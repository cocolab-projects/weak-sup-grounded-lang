import os
import sys
import numpy as np
import collections
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal

from utils import (AverageMeter, score_txt_logits, _reparameterize,
                    loss_multimodal, _log_mean_exp, gaussian_log_pdf, isotropic_gaussian_log_pdf,
                    bernoulli_log_pdf, get_text, get_image_text_joint_nll, get_image_text_joint_nll_cond_only)
from models import (TextEmbedding, TextEncoder, TextDecoder,
                    ColorEncoder, MultimodalEncoder, ColorDecoder, Finetune_Refgame)
from color_dataset import (ColorDataset, Colors_ReferenceGame)

SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
# N_SAMPLE = 1000

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir', type=str, help='where to load checkpoints from')
    parser.add_argument('out_dir', type=str, help='where to store results from')
    parser.add_argument('--sup_lvl', type=float, default=1.0,
                        help='supervision level, if any')
    parser.add_argument('--num_iter', type=int, default=1,
                        help='number of total iterations performed on each setting [default: 1]')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='lambda argument for text loss')
    parser.add_argument('--beta', type=float, default=10.0,
                        help='lambda argument for rgb loss')
    parser.add_argument('--context_condition', type=str, default='far',
                        help='whether the dataset is to include all data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    args = parser.parse_args()

    # set learning device
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.isdir(args.out_dir):
        print("Creating new folder... : {}".format(args.out_dir))
        os.makedirs(args.out_dir)

    def test_refgame_accuracy(vocab):
        '''
        Final accuracy: test on reference game dataset
        '''
        assert vocab != None
        print("Computing final accuracy for reference game settings...")

        ref_dataset = Colors_ReferenceGame(vocab, split='Test')
        ref_loader = DataLoader(ref_dataset, shuffle=False, batch_size=100)

        sup_finetune.eval()
        vae_txt_enc.eval()
        vae_rgb_enc.eval()
        vae_emb.eval()
        vae_mult_enc.eval()

        with torch.no_grad():
            total_count = 0
            correct_count = 0
            for batch_idx, (tgt_rgb, d1_rgb, d2_rgb, x_inp, x_tgt, x_len) in enumerate(ref_loader):
                batch_size = x_inp.size(0)
                tgt_rgb = tgt_rgb.to(device).float()
                d1_rgb = d1_rgb.to(device).float()
                d2_rgb = d2_rgb.to(device).float()
                x_inp = x_inp.to(device)
                x_len = x_len.to(device)

                # obtain embeddings
                z_x, _ = vae_txt_enc(x_inp, x_len)
                z_y, _ = vae_rgb_enc(tgt_rgb)
                z_d1, _ = vae_rgb_enc(d1_rgb)
                z_d2, _ = vae_rgb_enc(d2_rgb)
                # z_xy, _ = vae_mult_enc(y_rgb, x_src, x_len)

                # obtain predicted compatibility score
                tgt_score = sup_finetune(z_x, z_y)
                d1_score = sup_finetune(z_x, z_d1)
                d2_score = sup_finetune(z_x, z_d2)

                soft = nn.Softmax(dim=1)
                loss = soft(torch.cat([tgt_score,d1_score,d2_score], 1))
                softList = torch.argmax(loss, dim=1)

                correct_count += torch.sum(softList == 0).item()
                total_count += softList.size(0)

            accuracy = correct_count / float(total_count) * 100
            print('====> Final Accuracy: {}/{} = {}%'.format(correct_count, total_count, accuracy))
        return accuracy

    def load_finetune_checkpoint(folder='./', filename='model_best'):
        print("\nloading checkpoint file: {}.pth.tar ...\n".format(filename)) 
        checkpoint = torch.load(os.path.join(folder, filename + '.pth.tar'))
        epoch = checkpoint['epoch']
        vae_emb_sd = checkpoint['vae_emb']
        vae_rgb_enc_sd = checkpoint['vae_rgb_enc']
        vae_txt_enc_sd = checkpoint['vae_txt_enc']
        vae_mult_enc_sd = checkpoint['vae_mult_enc']
        sup_finetune_sd = checkpoint['sup_finetune']
        vocab = checkpoint['vocab']
        vocab_size = checkpoint['vocab_size']
        args = checkpoint['cmd_line_args']

        w2i = vocab['w2i']
        pad_index = w2i[PAD_TOKEN]

        vae_emb = TextEmbedding(vocab_size)
        vae_rgb_enc = ColorEncoder(args.z_dim)
        vae_txt_enc = TextEncoder(vae_emb, args.z_dim)
        vae_mult_enc = MultimodalEncoder(vae_emb, args.z_dim)

        vae_emb.load_state_dict(vae_emb_sd)
        vae_rgb_enc.load_state_dict(vae_rgb_enc_sd)
        vae_txt_enc.load_state_dict(vae_txt_enc_sd)
        vae_mult_enc.load_state_dict(vae_mult_enc_sd)

        sup_finetune = Finetune_Refgame(z_dim=args.z_dim)
        sup_finetune.load_state_dict(sup_finetune_sd)

        return epoch, args, vae_emb, vae_rgb_enc, vae_txt_enc, vae_mult_enc, sup_finetune, vocab, vocab_size

    print("=== begin testing ===")

    losses, mean_accuracies, sample_accuracies, cond_accuracies, diverge_rates, best_epochs = [], [], [], [], [], []
    for iter_num in range(1, args.num_iter + 1):
        filename = 'checkpoint_vae_{}_{}_alpha={}_beta={}_best'.format(args.sup_lvl,
                                                                        iter_num,
                                                                        args.alpha,
                                                                        args.beta)
        
        epoch, train_args, vae_emb, vae_rgb_enc, vae_txt_enc, vae_mult_enc, sup_finetune, vocab, vocab_size, = \
                                                        load_finetune_checkpoint(folder=args.load_dir, filename=filename)

        vae_emb.to(device)
        vae_rgb_enc.to(device)
        vae_txt_enc.to(device)
        vae_mult_enc.to(device)
        sup_finetune.to(device)

        print("iteration {} with alpha {} and beta {}\n".format(iter_num, train_args.alpha, train_args.beta))
        print("best training epoch: {}".format(epoch))

        # compute test loss & reference game accuracy
        cond_acc = test_refgame_accuracy(vocab)
        
        # diverge_rates.append(diverge_rate)
        # mean_accuracies.append(mean_acc)
        # sample_accuracies.append(sample_acc)
        cond_accuracies.append(cond_acc)
        best_epochs.append(epoch)

    # losses = np.array(losses)
    # mean_accuracies = np.array(mean_accuracies)
    # sample_accuracies = np.array(sample_accuracies)
    cond_accuracies = np.array(cond_accuracies)
    # diverge_rates = np.array(diverge_rates)

    # save files as np arrays
    print("\nsaving file to {} ...".format(args.out_dir))
    # np.save(os.path.join(args.out_dir, 'sample_accuracies_{}_alpha={}_beta={}.npy'.format(args.sup_lvl, args.alpha, args.beta)), sample_accuracies)
    # np.save(os.path.join(args.out_dir, 'mean_accuracies_{}_alpha={}_beta={}.npy'.format(args.sup_lvl, args.alpha, args.beta)), mean_accuracies)
    np.save(os.path.join(args.out_dir, 'cond_accuracies_{}_alpha={}_beta={}.npy'.format(args.sup_lvl, args.alpha, args.beta)), cond_accuracies)
    # np.save(os.path.join(args.out_dir, 'divergence_rates_{}_alpha={}_beta={}.npy'.format(args.sup_lvl, args.alpha, args.beta)), diverge_rates)
    print("... saving complete.")

    print("\n======> Best epochs: {}".format(best_epochs))
    print("======> Average conditional-based accuracy: {:4f}".format(np.mean(cond_accuracies)))

    print(args)

# def sanity_check(split='Test'):

#         print("performing sanity checks ...")

#         vae_emb.eval()
#         vae_rgb_enc.eval()
#         vae_txt_enc.eval()
#         vae_mult_enc.eval()
#         vae_rgb_dec.eval()
#         vae_txt_dec.eval()

#         example = torch.tensor([[70., 200., 70.],
#                                 [36, 220, 36],
#                                 [180, 50, 180],
#                                 [150, 150, 150],
#                                 [30, 30, 30],
#                                 [190, 30, 30]
#                                 ]).to(device)
#         z_y_mu, z_y_logvar = vae_rgb_enc(example)
#         y_mu_z_y = vae_rgb_dec(z_y_mu)
#         print("encoded colors: {}".format(example))
#         print("decoded colors: {}\n".format(y_mu_z_y * 255.0))

#         example = torch.tensor([[528, 13],
#                                 [528, 0],
#                                 [528, 7]]
#                                 )
#         z_x_mu, z_x_logvar = vae_txt_enc(example.to(device), torch.LongTensor([2,2,2]))
#         decoded_colors = vae_rgb_dec(z_x_mu)
#         sampled_colors = vae_rgb_dec(torch.randn_like(z_x_mu) * torch.exp(0.5 * z_x_logvar) + z_x_mu)
#         print("encoded tests: {}".format(example))
#         print("decoded colors: {}\n".format(decoded_colors))
#         print("sampled colors: {}\n".format(sampled_colors))

#         return