import os
import sys
import numpy as np
from tqdm import tqdm

import torch 
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal

from utils import (AverageMeter, score_txt_logits, reparameterize,
                    loss_multimodal, log_mean_exp, gaussian_log_pdf, isotropic_gaussian_log_pdf,
                    bernoulli_log_pdf, get_text, get_image_text_joint_nll)
from models import (TextEmbedding, TextEncoder, TextDecoder,
                    ColorEncoder, MultimodalEncoder, ColorDecoder)
from color_dataset import (ColorDataset, Colors_ReferenceGame)

SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

N_SAMPLE = 80

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir', type=str, help='where to load checkpoints from')
    parser.add_argument('out_dir', type=str, help='where to store results from')
    parser.add_argument('--sup_lvl', type=float, help='supervision level, if any')
    parser.add_argument('--num_iter', type=int, default=1,
                        help='number of total iterations performed on each setting [default: 1]')
    parser.add_argument('--hard', action='store_true', help='whether the dataset is to be easy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    args = parser.parse_args()

    # set learning device
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    def test_loss(split='Test'):
        '''
        Test model on newly seen dataset -- gives final test loss
        '''
        assert vocab != None
        print("Computing final test loss on newly seen dataset...")

        test_dataset = ColorDataset(vocab=vocab, split=split, hard=args.hard)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=100)

        vae_emb.eval()
        vae_rgb_enc.eval()
        vae_txt_enc.eval()
        vae_mult_enc.eval()
        vae_rgb_dec.eval()
        vae_txt_dec.eval()

        with torch.no_grad():
            loss_meter = AverageMeter()
            for batch_idx, (y_rgb, x_src, x_tgt, x_len) in enumerate(test_loader):
                batch_size = x_src.size(0) 
                y_rgb = y_rgb.to(device).float()
                x_src = x_src.to(device)
                x_tgt = x_tgt.to(device)
                x_len = x_len.to(device)

                # Encode to |z|
                z_x_mu, z_x_logvar = vae_txt_enc(x_src, x_len)
                z_y_mu, z_y_logvar = vae_rgb_enc(y_rgb)
                z_xy_mu, z_xy_logvar = vae_mult_enc(y_rgb, x_src, x_len)

                # sample via reparametrization
                z_sample_x = reparameterize(z_x_mu, z_x_logvar)
                z_sample_y = reparameterize(z_y_mu, z_y_logvar)
                z_sample_xy = reparameterize(z_xy_mu, z_xy_logvar)

                # "predictions"
                y_mu_z_y = vae_rgb_dec(z_sample_y)
                y_mu_z_xy = vae_rgb_dec(z_sample_xy)
                x_logit_z_x = vae_txt_dec(z_sample_x, x_src, x_len)
                x_logit_z_xy = vae_txt_dec(z_sample_xy, x_src, x_len)

                out = {'z_x_mu': z_x_mu, 'z_x_logvar': z_x_logvar,
                        'z_y_mu': z_y_mu, 'z_y_logvar': z_y_logvar,
                        'z_xy_mu': z_xy_mu, 'z_xy_logvar': z_xy_logvar,
                        'y_mu_z_y': y_mu_z_y, 'y_mu_z_xy': y_mu_z_xy, 
                        'x_logit_z_x': x_logit_z_x, 'x_logit_z_xy': x_logit_z_xy,
                        'y': y_rgb, 'x': x_tgt, 'pad_index': pad_index}

                # compute loss
                loss = loss_multimodal(out, batch_size, alpha=train_args.alpha, beta=train_args.beta)
                loss_meter.update(loss.item(), batch_size)

            print('====> Final Test Loss: {:.4f}'.format(loss_meter.avg))
        return loss_meter.avg

    def test_refgame_accuracy():
        '''
        Final accuracy: test on reference game dataset
        '''
        print("Computing final accuracy for reference game settings...")

        ref_dataset = Colors_ReferenceGame(vocab, split='Test', hard=args.hard)
        ref_loader = DataLoader(ref_dataset, shuffle=False, batch_size=100)

        vae_emb.eval()
        vae_rgb_enc.eval()
        vae_txt_enc.eval()
        vae_mult_enc.eval()
        vae_rgb_dec.eval()
        vae_txt_dec.eval()

        with torch.no_grad():
            loss_meter = AverageMeter()

            total_count = 0
            correct_count = 0

            with tqdm(total=len(ref_loader)) as pbar:
                for batch_idx, (y_rgb, d1_rgb, d2_rgb, x_src, x_tgt, x_len) in enumerate(ref_loader):
                    batch_size = x_src.size(0) 
                    y_rgb = y_rgb.to(device).float()
                    d1_rgb = d1_rgb.to(device).float()
                    d2_rgb = d2_rgb.to(device).float()
                    x_src = x_src.to(device)
                    x_tgt = x_tgt.to(device)
                    x_len = x_len.to(device)

                    # Encode to |z|
                    z_x_mu, z_x_logvar = vae_txt_enc(x_src, x_len)
                    z_y_mu, z_y_logvar = vae_rgb_enc(y_rgb)
                    z_xy_mu, z_xy_logvar = vae_mult_enc(y_rgb, x_src, x_len)

                    # sample and obtain expected value
                    for i in range(batch_size):
                        z_samples = torch.randn(N_SAMPLE, train_args.z_dim).to(device) * torch.exp(0.5 * z_xy_logvar[i]) + z_xy_mu[i]
                        y_mu_list = vae_rgb_dec(z_samples)
                        # print(get_text(vocab['i2w'], x_tgt[i], x_len[i]))
                        # print(y_mu_list[0] * 255.)

                        x_tgt_logits_list = vae_txt_dec(z_samples, x_src[i].unsqueeze(0).repeat(N_SAMPLE, 1),
                                                                    x_len[i].unsqueeze(0).repeat(N_SAMPLE))
                        elt_max_len = x_tgt_logits_list.size(1)
                        x_tgt_i = x_tgt[i, : elt_max_len]
                        x_len_i = elt_max_len

                        # "predictions"
                        p_x_y1 = get_image_text_joint_nll(y_rgb[i], y_mu_list, x_tgt_i, x_tgt_logits_list, z_samples, z_xy_mu[i], z_xy_logvar[i], pad_index)
                        p_x_y2 = get_image_text_joint_nll(d1_rgb[i], y_mu_list, x_tgt_i, x_tgt_logits_list, z_samples, z_xy_mu[i], z_xy_logvar[i], pad_index)
                        p_x_y3 = get_image_text_joint_nll(d2_rgb[i], y_mu_list, x_tgt_i, x_tgt_logits_list, z_samples, z_xy_mu[i], z_xy_logvar[i], pad_index)

                        total_count += 1
                        correct = False
                        if p_x_y1 < p_x_y2 and p_x_y1 < p_x_y3:
                            correct_count += 1
                            correct = True
                        if (i % 50 == 0):
                            match_text = get_text(vocab['i2w'], x_tgt_i, x_len_i)
                            print("color: {} <===> target text: {} <====> {}".format(y_rgb[i] * 255.0, match_text, x_tgt_i))
                            print("mean color prediction: {} with std {}".format(torch.mean(y_mu_list, dim=0), torch.std(y_mu_list, dim=0)))
                            print("correct? {} ======> p(x,y1): {:4f} p(x,y2): {:4f} p(x,y3): {:4f}".format('T' if correct else 'F', p_x_y1, p_x_y2, p_x_y3))
                            print()
                    pbar.update()

            accuracy = correct_count / float(total_count) * 100
            print('====> Final Accuracy: {}/{} = {}%'.format(correct_count, total_count, accuracy))
            print()
        return accuracy

    def load_checkpoint(folder='./', filename='model_best'):
        checkpoint = torch.load(folder + filename + '.pth.tar')
        epoch = checkpoint['epoch']
        vae_emb_sd = checkpoint['vae_emb']
        vae_rgb_enc_sd = checkpoint['vae_rgb_enc']
        vae_txt_enc_sd = checkpoint['vae_txt_enc']
        vae_mult_enc_sd = checkpoint['vae_mult_enc']
        vae_rgb_dec_sd = checkpoint['vae_rgb_dec']
        vae_txt_dec_sd = checkpoint['vae_txt_dec']
        vocab = checkpoint['vocab']
        vocab_size = checkpoint['vocab_size']
        args = checkpoint['cmd_line_args']

        w2i = vocab['w2i']
        pad_index = w2i[PAD_TOKEN]

        vae_emb = TextEmbedding(vocab_size)
        vae_rgb_enc = ColorEncoder(args.z_dim)
        vae_txt_enc = TextEncoder(vae_emb, args.z_dim)
        vae_mult_enc = MultimodalEncoder(vae_emb, args.z_dim)
        vae_rgb_dec = ColorDecoder(args.z_dim)
        vae_txt_dec = TextDecoder(vae_emb, args.z_dim, w2i[SOS_TOKEN], w2i[EOS_TOKEN],
                                    w2i[PAD_TOKEN], w2i[UNK_TOKEN], word_dropout=args.dropout)
        vae_emb.load_state_dict(vae_emb_sd)
        vae_rgb_enc.load_state_dict(vae_rgb_enc_sd)
        vae_txt_enc.load_state_dict(vae_txt_enc_sd)
        vae_mult_enc.load_state_dict(vae_mult_enc_sd)
        vae_rgb_dec.load_state_dict(vae_rgb_dec_sd)
        vae_txt_dec.load_state_dict(vae_txt_dec_sd)

        return epoch, args, vae_emb, vae_rgb_enc, vae_txt_enc, vae_mult_enc, vae_rgb_dec, vae_txt_dec, vocab, vocab_size, pad_index

    def sanity_check(split='Test'):
        vae_emb.eval()
        vae_rgb_enc.eval()
        vae_txt_enc.eval()
        vae_mult_enc.eval()
        vae_rgb_dec.eval()
        vae_txt_dec.eval()

        z_y_mu, z_y_logvar = vae_rgb_enc(torch.tensor(
                                        [[70., 200., 70.],
                                        [36, 220, 36],
                                        [180, 50, 180],
                                        [150, 150, 150],
                                        [30, 30, 30],
                                        [190, 30, 30]
                                        ]).to(device))
        y_mu_z_y = vae_rgb_dec(z_y_mu)
        print(y_mu_z_y * 255.0)
        return

    print("=== begin testing ===")

    losses, accuracies = [], []
    for iter_num in range(1, args.num_iter + 1):

        print("loading checkpoint files ...")
        epoch, train_args, vae_emb, vae_rgb_enc, vae_txt_enc, vae_mult_enc, vae_rgb_dec, vae_txt_dec, vocab, vocab_size, pad_index = \
            load_checkpoint(folder=args.load_dir,
                            filename='checkpoint_{}_{}_best'.format(args.sup_lvl, iter_num))
        
        vae_emb.to(device)
        vae_rgb_enc.to(device)
        vae_txt_enc.to(device)
        vae_mult_enc.to(device)
        vae_rgb_dec.to(device)
        vae_txt_dec.to(device)

        print("... loading complete.")

        print("performing sanity checks ...")
        sanity_check()
        print()

        print()
        print("iteration {} with alpha {} and beta {}".format(iter_num, train_args.alpha, train_args.beta))
        print("best training epoch: {}".format(epoch))

        losses.append(test_loss())
        accuracies.append(test_refgame_accuracy())

    losses = np.array(losses)
    accuracies = np.array(accuracies)
    np.save(os.path.join(args.out_dir, 'accuracies_{}.npy'.format(args.sup_lvl)), accuracies)
    # np.save(os.path.join(args.out_dir, 'final_losses_{}.npy'.format(args.sup_lvl)), losses)

    print()
    print("======> Average loss: {:6f}".format(np.mean(losses)))
    print("======> Average accuracy: {:4f}".format(np.mean(accuracies)))