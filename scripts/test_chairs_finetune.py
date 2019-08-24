import os
import sys
import numpy as np
import collections
from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from utils import (AverageMeter, score_txt_logits, _reparameterize,
                    loss_multimodal, _log_mean_exp, gaussian_log_pdf, isotropic_gaussian_log_pdf,
                    bernoulli_log_pdf, get_text, get_image_text_joint_nll)
from models import (TextEmbedding, TextEncoder, TextDecoder,
                    ImageEncoder, ImageTextEncoder, ImageDecoder, Finetune_Refgame)

SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
N_SAMPLE = 150

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir', type=str, help='where to load checkpoints from')
    parser.add_argument('out_dir', type=str, help='where to store results from')
    parser.add_argument('--dataset', type=str, default='chairs')
    parser.add_argument('--sup_lvl', type=float, default=1.0,
                        help='supervision level, if any')
    parser.add_argument('--num_iter', type=int, default=1,
                        help='number of total iterations performed on each setting [default: 1]')
    parser.add_argument('--alpha', type=float, default=1,
                        help='lambda argument for text loss')
    parser.add_argument('--beta', type=float, default=1,
                        help='lambda argument for image loss')
    parser.add_argument('--z_dim', type=int, default=100,
                        help='number of latent dimension [default = 100]')
    parser.add_argument('--context_condition', type=str, default='far',
                        help='whether the dataset is to include all data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    args = parser.parse_args()

    print("Called python script: test_chairs_finetune.py")
    print(args)

    if args.dataset == 'chairs':
        from chair_dataset import (Chairs_ReferenceGame, Weaksup_Chairs_Reference)
    # if args.dataset == 'critters':
    #     from critter_dataset import (Critters_ReferenceGame, Weaksup_Critters_Reference)

    # set learning device
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Define latent dimension |z_dim|
    z_dim = args.z_dim

    if not os.path.isdir(args.out_dir):
        print("Creating new folder... : {}".format(args.out_dir))
        os.makedirs(args.out_dir)

    def test_refgame_accuracy(split='Test'):
        """Function: test_refgame_accuracy
        Returns:
            (float) mean_acc: mean-based choice accuracy
            (float) sample_acc: sample-based choice accuracy
            (float) diverge_rate: choice divergence between mean-based and sample-based choice
        Compute final accuracy test on reference game dataset
        """
        print("Computing final accuracy for reference game settings...")

        sup_finetune.eval()
        vae_txt_enc.eval()
        vae_img_enc.eval()
        vae_emb.eval()
        vae_mult_enc.eval()

        with torch.no_grad():
            loss_meter = AverageMeter()
            total_count = 0
            correct_count = 0

            with tqdm(total=len(test_loader)) as pbar:
                for batch_idx, (tgt_img, d1_img, d2_img, x_inp, x_tgt, x_len) in enumerate(test_loader):
                    batch_size = x_inp.size(0)
                    tgt_img = tgt_img.to(device).float()
                    d1_img = d1_img.to(device).float()
                    d2_img = d2_img.to(device).float()
                    x_inp = x_inp.to(device)
                    x_len = x_len.to(device)

                    # obtain embeddings
                    z_x, _ = vae_txt_enc(x_inp, x_len)
                    z_y, _ = vae_img_enc(tgt_img)
                    z_d1, _ = vae_img_enc(d1_img)
                    z_d2, _ = vae_img_enc(d2_img)
                    # z_xy, _ = vae_mult_enc(y_img, x_src, x_len)

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
        vae_img_enc_sd = checkpoint['vae_img_enc']
        vae_txt_enc_sd = checkpoint['vae_txt_enc']
        vae_mult_enc_sd = checkpoint['vae_mult_enc']
        sup_finetune_sd = checkpoint['sup_finetune']
        vocab = checkpoint['vocab']
        vocab_size = checkpoint['vocab_size']
        args = checkpoint['cmd_line_args']

        w2i = vocab['w2i']
        pad_index = w2i[PAD_TOKEN]

        channels, img_size = 3, 32
        vae_emb = TextEmbedding(vocab_size)
        vae_img_enc = ImageEncoder(channels, img_size, z_dim)
        vae_txt_enc = TextEncoder(vae_emb, z_dim)
        vae_mult_enc = ImageTextEncoder(channels, img_size, z_dim, vae_emb)

        vae_emb.load_state_dict(vae_emb_sd)
        vae_img_enc.load_state_dict(vae_img_enc_sd)
        vae_txt_enc.load_state_dict(vae_txt_enc_sd)
        vae_mult_enc.load_state_dict(vae_mult_enc_sd)

        sup_finetune = Finetune_Refgame(z_dim=args.z_dim)
        sup_finetune.load_state_dict(sup_finetune_sd)

        return epoch, args, vae_emb, vae_img_enc, vae_txt_enc, vae_mult_enc, sup_finetune, vocab, vocab_size

    print("=== begin testing ===")

    losses, mean_accuracies, sample_accuracies, cond_accuracies, best_epochs = [], [], [], [], []
    for iter_num in range(1, args.num_iter + 1):
        filename = 'checkpoint_vae_{}_{}_alpha={}_beta={}_best'.format(args.sup_lvl,
                                                                        iter_num,
                                                                        args.alpha,
                                                                        args.beta)
        
        epoch, train_args, vae_emb, vae_img_enc, vae_txt_enc, vae_mult_enc, sup_finetune, vocab, vocab_size = \
                        load_finetune_checkpoint(folder=args.load_dir, filename=filename)
        
        vae_emb.to(device)
        vae_img_enc.to(device)
        vae_txt_enc.to(device)
        vae_mult_enc.to(device)
        sup_finetune.to(device)

        # sanity check
        print("iteration {} with alpha {} and beta {}\n".format(iter_num, train_args.alpha, train_args.beta))
        print("best training epoch: {}".format(epoch))

        if args.dataset == 'chairs':
            test_dataset = Chairs_ReferenceGame(vocab=vocab, split='Test', context_condition=args.context_condition)
        # if args.dataset == 'critters':
        #     image_size = 32
        #     image_transform = transforms.Compose([
        #                                             transforms.Resize(image_size),
        #                                             transforms.CenterCrop(image_size),
        #                                         ])
        #     test_dataset = Critters_ReferenceGame(vocab=vocab, split='Validation', context_condition=args.context_condition, image_transform=image_transform)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=100, num_workers=8)
        # compute test loss & reference game accuracy
        # losses.append(test_loss())
        cond_acc = test_refgame_accuracy()
        
        # mean_accuracies.append(mean_acc)
        # sample_accuracies.append(sample_acc)
        cond_accuracies.append(cond_acc)
        best_epochs.append(epoch)

    # losses = np.array(losses)
    # mean_accuracies = np.array(mean_accuracies)
    # sample_accuracies = np.array(sample_accuracies)
    cond_accuracies = np.array(cond_accuracies)

    # save files as np arrays
    print("saving file to {} ...".format(args.out_dir))
    # np.save(os.path.join(args.out_dir, 'sample_accuracies_{}_alpha={}_beta={}.npy'.format(args.sup_lvl, args.alpha, args.beta)), sample_accuracies)
    # np.save(os.path.join(args.out_dir, 'mean_accuracies_{}_alpha={}_beta={}.npy'.format(args.sup_lvl, args.alpha, args.beta)), mean_accuracies)
    np.save(os.path.join(args.out_dir, 'cond_accuracies_{}_alpha={}_beta={}.npy'.format(args.sup_lvl, args.alpha, args.beta)), cond_accuracies)
    print("... saving complete.")

    print("\n======> Best epochs: {}".format(best_epochs))
    print("======> Average conditional accuracy: {:4f}".format(np.mean(cond_accuracies)))

    print(args)



    # def get_sampled_joint_prob(y_i, x_src, x_tgt, x_len, z_xy_mu, z_xy_logvar):
    #     z_samples = torch.randn(N_SAMPLE, train_args.z_dim).to(device) * torch.exp(0.5 * z_xy_logvar) + z_xy_mu

    #     y_mu_list = vae_img_dec(z_samples)
    #     x_tgt_logits_list = vae_txt_dec(z_samples, x_src.unsqueeze(0).repeat(N_SAMPLE, 1),
    #                                                 x_len.unsqueeze(0).repeat(N_SAMPLE))
    #     elt_max_len = x_tgt_logits_list.size(1)
    #     x_tgt_i = x_tgt[:elt_max_len]
    #     x_len_i = elt_max_len

    #     return get_image_text_joint_nll(y_i, y_mu_list, x_tgt_i, x_tgt_logits_list, x_len, z_samples, z_xy_mu, z_xy_logvar, pad_index)

    # def get_mean_joint_prob(y, x_src, x_tgt, x_len, z_xy_mu, z_xy_logvar, verbose=False):
    #     y_mu_z_xy = vae_img_dec(z_xy_mu.unsqueeze(0))
    #     x_tgt_logits = vae_txt_dec(z_xy_mu.unsqueeze(0), x_src.unsqueeze(0), x_len.unsqueeze(0))
    #     x_tgt = x_tgt[:x_len]

    #     return get_image_text_joint_nll(y, y_mu_z_xy, x_tgt, x_tgt_logits, x_len, z_xy_mu.unsqueeze(0), z_xy_mu, z_xy_logvar, pad_index, verbose)

    # # mean-based estimator of joint probabilities based on z ~ q(z|x,y)
    # p_x_y1_sampled = get_sampled_joint_prob(tgt_img[i], x_src[i], x_tgt[i], x_len[i], z_xy_mu_tgt[i], z_xy_logvar_tgt[i])
    # p_x_y2_sampled = get_sampled_joint_prob(d1_img[i], x_src[i], x_tgt[i], x_len[i], z_xy_mu_d1[i], z_xy_logvar_d1[i])
    # p_x_y3_sampled = get_sampled_joint_prob(d2_img[i], x_src[i], x_tgt[i], x_len[i], z_xy_mu_d2[i], z_xy_logvar_d2[i])

    # # sample-based estimator of joint probabilities based on z ~ q(z|x,y)
    # p_x_y1_mean = get_mean_joint_prob(tgt_img[i], x_src[i], x_tgt[i], x_len[i], z_xy_mu_tgt[i], z_xy_logvar_tgt[i], verbose=verbose)
    # p_x_y2_mean = get_mean_joint_prob(d1_img[i], x_src[i], x_tgt[i], x_len[i], z_xy_mu_d1[i], z_xy_logvar_d1[i], verbose=verbose)
    # p_x_y3_mean = get_mean_joint_prob(d2_img[i], x_src[i], x_tgt[i], x_len[i], z_xy_mu_d2[i], z_xy_logvar_d2[i], verbose=verbose)

    # choice based on conditional distribution z ~ q(z|x)
    # if p_x_y1_mean < p_x_y2_mean and p_x_y1_mean < p_x_y3_mean:
    #     mean_correct_count += 1
    #     mean_correct = True
    # if p_x_y1_sampled < p_x_y2_sampled and p_x_y1_sampled < p_x_y3_sampled:
    #     sample_correct_count += 1
    #     sample_correct = True