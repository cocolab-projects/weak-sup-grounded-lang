import os
import sys
import numpy as np
import collections
from tqdm import tqdm

import torch 
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal

from utils import (AverageMeter, score_txt_logits, _reparameterize,
                    loss_multimodal, _log_mean_exp, gaussian_log_pdf, isotropic_gaussian_log_pdf,
                    bernoulli_log_pdf, get_text, get_image_text_joint_nll)
from models import (TextEmbedding, TextEncoder, TextDecoder,
                    ImageEncoder, ImageTextEncoder, ImageDecoder)
from chair_dataset import (Chairs_ReferenceGame)

SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
N_SAMPLE = 100

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir', type=str, help='where to load checkpoints from')
    parser.add_argument('out_dir', type=str, help='where to store results from')
    parser.add_argument('--sup_lvl', type=float, default=1.0,
                        help='supervision level, if any')
    parser.add_argument('--num_iter', type=int, default=1,
                        help='number of total iterations performed on each setting [default: 1]')
    parser.add_argument('--alpha', type=float, default=1,
                        help='lambda argument for text loss')
    parser.add_argument('--beta', type=float, default=1,
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

    def test_loss(split='Test'):
        assert vocab != None
        print("Computing final test loss on newly seen dataset...")

        vae_emb.eval()
        vae_img_enc.eval()
        vae_txt_enc.eval()
        vae_mult_enc.eval()
        vae_img_dec.eval()
        vae_txt_dec.eval()

        with torch.no_grad():
            loss_meter = AverageMeter()
            pbar = tqdm(total=len(test_loader))
            for batch_idx, (tgt_chair, d1_chair, d2_chair, x_tgt, x_src, x_len) in enumerate(test_loader):
                batch_size = x_src.size(0) 
                tgt_chair = tgt_chair.to(device).float()
                x_src = x_src.to(device)
                x_tgt = x_tgt.to(device)
                x_len = x_len.to(device)

                # Encode to |z|
                z_x_mu, z_x_logvar = vae_txt_enc(x_src, x_len)
                z_y_mu, z_y_logvar = vae_img_enc(tgt_chair)
                z_xy_mu, z_xy_logvar = vae_mult_enc(tgt_chair, x_src, x_len)

                # sample via reparametrization
                z_sample_x = _reparameterize(z_x_mu, z_x_logvar)
                z_sample_y = _reparameterize(z_y_mu, z_y_logvar)
                z_sample_xy = _reparameterize(z_xy_mu, z_xy_logvar)

                # "predictions"
                y_mu_z_y = vae_img_dec(z_sample_y)
                y_mu_z_xy = vae_img_dec(z_sample_xy)
                x_logit_z_x = vae_txt_dec(z_sample_x, x_src, x_len)
                x_logit_z_xy = vae_txt_dec(z_sample_xy, x_src, x_len)

                out = {'z_x_mu': z_x_mu, 'z_x_logvar': z_x_logvar,
                        'z_y_mu': z_y_mu, 'z_y_logvar': z_y_logvar,
                        'z_xy_mu': z_xy_mu, 'z_xy_logvar': z_xy_logvar,
                        'y_mu_z_y': y_mu_z_y, 'y_mu_z_xy': y_mu_z_xy, 
                        'x_logit_z_x': x_logit_z_x, 'x_logit_z_xy': x_logit_z_xy,
                        'y': tgt_chair, 'x': x_tgt, 'x_len': x_len, 'pad_index': pad_index}

                # compute loss
                loss = loss_multimodal(out, batch_size, alpha=train_args.alpha, beta=train_args.beta)
                loss_meter.update(loss.item(), batch_size)
                pbar.set_postfix({'loss': loss_meter.avg})
                pbar.update()
            pbar.close()

            print('====> Final Test Loss: {:.4f}'.format(loss_meter.avg))
        return loss_meter.avg

    def get_sampled_joint_prob(y_i, x_src, x_tgt, x_len, z_xy_mu, z_xy_logvar):
        z_samples = torch.randn(N_SAMPLE, train_args.z_dim).to(device) * torch.exp(0.5 * z_xy_logvar) + z_xy_mu

        y_mu_list = vae_img_dec(z_samples)
        x_tgt_logits_list = vae_txt_dec(z_samples, x_src.unsqueeze(0).repeat(N_SAMPLE, 1),
                                                    x_len.unsqueeze(0).repeat(N_SAMPLE))
        elt_max_len = x_tgt_logits_list.size(1)
        x_tgt_i = x_tgt[:elt_max_len]
        x_len_i = elt_max_len

        return get_image_text_joint_nll(y_i, y_mu_list, x_tgt_i, x_tgt_logits_list, x_len, z_samples, z_xy_mu, z_xy_logvar, pad_index)

    def get_mean_joint_prob(y, x_src, x_tgt, x_len, z_xy_mu, z_xy_logvar, verbose=False):
        y_mu_z_xy = vae_img_dec(z_xy_mu.unsqueeze(0))
        x_tgt_logits = vae_txt_dec(z_xy_mu.unsqueeze(0), x_src.unsqueeze(0), x_len.unsqueeze(0))
        x_tgt = x_tgt[:x_len]

        return get_image_text_joint_nll(y, y_mu_z_xy, x_tgt, x_tgt_logits, x_len, z_xy_mu.unsqueeze(0), z_xy_mu, z_xy_logvar, pad_index, verbose)

    def get_conditional_choice(y_1, y_2, y_3, z_mu):
        pred_img_cond = vae_img_dec(z_mu)
        diff_tgt = bernoulli_log_pdf(y_1.view(-1).unsqueeze(0), pred_img_cond.view(-1))
        diff_d1 = bernoulli_log_pdf(y_2.view(-1).unsqueeze(0), pred_img_cond.view(-1))
        diff_d2 = bernoulli_log_pdf(y_3.view(-1).unsqueeze(0), pred_img_cond.view(-1))
        return pred_img_cond, torch.argmax(torch.Tensor([diff_tgt, diff_d1, diff_d2]))

    def test_refgame_accuracy(split='Test'):
        """Function: test_refgame_accuracy
        Returns:
            (float) mean_acc: mean-based choice accuracy
            (float) sample_acc: sample-based choice accuracy
            (float) diverge_rate: choice divergence between mean-based and sample-based choice
        Compute final accuracy test on reference game dataset
        """
        print("Computing final accuracy for reference game settings...")

        vae_emb.eval()
        vae_img_enc.eval()
        vae_txt_enc.eval()
        vae_mult_enc.eval()
        vae_img_dec.eval()
        vae_txt_dec.eval()

        with torch.no_grad():
            loss_meter = AverageMeter()

            total_count = 0
            mean_correct_count, sample_correct_count, cond_correct_count, diverge_count = 0, 0, 0, 0

            with tqdm(total=len(test_loader)) as pbar:
                for batch_idx, (tgt_chair, d1_chair, d2_chair, x_tgt, x_src, x_len) in enumerate(test_loader):
                    batch_size = x_src.size(0) 
                    tgt_chair = tgt_chair.to(device).float()
                    d1_chair = d1_chair.to(device).float()
                    d2_chair = d2_chair.to(device).float()
                    x_src = x_src.to(device)
                    x_tgt = x_tgt.to(device)
                    x_len = x_len.to(device)

                    # Encode to |z|
                    z_x_mu, z_x_logvar = vae_txt_enc(x_src, x_len)
                    z_xy_mu_tgt, z_xy_logvar_tgt = vae_mult_enc(tgt_chair, x_src, x_len)
                    z_xy_mu_d1, z_xy_logvar_d1 = vae_mult_enc(d1_chair, x_src, x_len)
                    z_xy_mu_d2, z_xy_logvar_d2 = vae_mult_enc(d2_chair, x_src, x_len)

                    # check accuracy for each datapoint via mean, sampling, conditional
                    for i in range(batch_size):
                        verbose = i % 25 == 0
                        total_count += 1

                        # mean-based estimator of joint probabilities based on z ~ q(z|x,y)
                        p_x_y1_sampled = get_sampled_joint_prob(tgt_chair[i], x_src[i], x_tgt[i], x_len[i], z_xy_mu_tgt[i], z_xy_logvar_tgt[i])
                        p_x_y2_sampled = get_sampled_joint_prob(d1_chair[i], x_src[i], x_tgt[i], x_len[i], z_xy_mu_d1[i], z_xy_logvar_d1[i])
                        p_x_y3_sampled = get_sampled_joint_prob(d2_chair[i], x_src[i], x_tgt[i], x_len[i], z_xy_mu_d2[i], z_xy_logvar_d2[i])

                        # sample-based estimator of joint probabilities based on z ~ q(z|x,y)
                        p_x_y1_mean = get_mean_joint_prob(tgt_chair[i], x_src[i], x_tgt[i], x_len[i], z_xy_mu_tgt[i], z_xy_logvar_tgt[i], verbose=verbose)
                        p_x_y2_mean = get_mean_joint_prob(d1_chair[i], x_src[i], x_tgt[i], x_len[i], z_xy_mu_d1[i], z_xy_logvar_d1[i], verbose=verbose)
                        p_x_y3_mean = get_mean_joint_prob(d2_chair[i], x_src[i], x_tgt[i], x_len[i], z_xy_mu_d2[i], z_xy_logvar_d2[i], verbose=verbose)

                        # choice based on conditional distribution z ~ q(z|x)
                        pred_rgb_cond, cond_choice = get_conditional_choice(tgt_chair[i], d1_chair[i], d2_chair[i], z_x_mu[i])
                        
                        mean_correct, sample_correct, cond_correct, diverge = False, False, False, False

                        if p_x_y1_mean < p_x_y2_mean and p_x_y1_mean < p_x_y3_mean:
                            mean_correct_count += 1
                            mean_correct = True
                        if p_x_y1_sampled < p_x_y2_sampled and p_x_y1_sampled < p_x_y3_sampled:
                            sample_correct_count += 1
                            sample_correct = True
                        if cond_choice == 1:
                            cond_correct_count += 1
                            cond_correct = True
                        if (sample_correct and not mean_correct) or (mean_correct and not sample_correct):
                            diverge_count += 1
                            diverge = True
                        if verbose:
                            match_text = get_text(vocab['i2w'], x_tgt[i], x_len[i])
                            print("================== ==================")
                            
                            print("mean-based choice correct? {}".format('T' if mean_correct else 'F'))
                            print("sample-based choice correct? {}".format('T' if sample_correct else 'F'))
                            print("conditional distribution-based choice correct? {}".format('T' if cond_correct else 'F'))
                            print("p_x_y1_sampled: {}, p_x_y2_sampled: {}, p_x_y3_sampled: {}".format(p_x_y1_sampled, p_x_y2_sampled, p_x_y3_sampled))
                            print("p_x_y1_mean: {}, p_x_y2_mean: {}, p_x_y3_mean: {}".format(p_x_y1_mean, p_x_y2_mean, p_x_y3_mean))
                            
                            print("\n total count currently: {}".format(total_count))
                            print("current mean-based choice accuracy: {}".format(mean_correct_count / total_count))
                            print("current sample-based choice accuracy: {}".format(sample_correct_count / total_count))
                            
                    pbar.update()

            mean_acc = mean_correct_count / float(total_count) * 100
            sample_acc = sample_correct_count / float(total_count) * 100
            cond_acc = cond_correct_count / float(total_count) * 100
            diverge_rate = diverge_count / float(total_count) * 100
            print('====> Final Sample-based Accuracy: {}/{} = {}%'.format(sample_correct_count, total_count, sample_acc))
            print('====> Final Mean-based Accuracy: {}/{} = {}%'.format(mean_correct_count, total_count, mean_acc))
            print('====> Final Conditional Accuracy: {}/{} = {}%'.format(cond_correct_count, total_count, cond_acc))
            print('====> Final Divergence Rate: {}/{} = {}%\n'.format(diverge_count, total_count, diverge_rate))
        return mean_acc, sample_acc, cond_acc, diverge_rate

    def load_checkpoint(folder='./', filename='model_best'):
        print("\nloading checkpoint file: {}.pth.tar ...\n".format(filename)) 
        checkpoint = torch.load(os.path.join(folder, filename + '.pth.tar'))
        epoch = checkpoint['epoch']
        vae_emb_sd = checkpoint['vae_emb']
        vae_img_enc_sd = checkpoint['vae_img_enc']
        vae_txt_enc_sd = checkpoint['vae_txt_enc']
        vae_mult_enc_sd = checkpoint['vae_mult_enc']
        vae_img_dec_sd = checkpoint['vae_img_dec']
        vae_txt_dec_sd = checkpoint['vae_txt_dec']
        vocab = checkpoint['vocab']
        vocab_size = checkpoint['vocab_size']
        args = checkpoint['cmd_line_args']

        w2i = vocab['w2i']
        pad_index = w2i[PAD_TOKEN]

        z_dim, channels, img_size = 100, 3, 32
        vae_emb = TextEmbedding(vocab_size)
        vae_img_enc = ImageEncoder(channels, img_size, z_dim)
        vae_txt_enc = TextEncoder(vae_emb, z_dim)
        vae_mult_enc = ImageTextEncoder(channels, img_size, z_dim, vae_emb)
        vae_img_dec = ImageDecoder(channels, img_size, z_dim)
        vae_txt_dec = TextDecoder(vae_emb, z_dim, w2i[SOS_TOKEN], w2i[EOS_TOKEN],
                                    w2i[PAD_TOKEN], w2i[UNK_TOKEN], word_dropout=args.dropout)

        vae_emb.load_state_dict(vae_emb_sd)
        vae_img_enc.load_state_dict(vae_img_enc_sd)
        vae_txt_enc.load_state_dict(vae_txt_enc_sd)
        vae_mult_enc.load_state_dict(vae_mult_enc_sd)
        vae_img_dec.load_state_dict(vae_img_dec_sd)
        vae_txt_dec.load_state_dict(vae_txt_dec_sd)

        return epoch, args, vae_emb, vae_img_enc, vae_txt_enc, vae_mult_enc, vae_img_dec, vae_txt_dec, vocab, vocab_size, pad_index

    def sanity_check(split='Test'):

        print("performing sanity checks ...")

        vae_emb.eval()
        vae_rgb_enc.eval()
        vae_txt_enc.eval()
        vae_mult_enc.eval()
        vae_rgb_dec.eval()
        vae_txt_dec.eval()

        example = torch.tensor([[70., 200., 70.],
                                [36, 220, 36],
                                [180, 50, 180],
                                [150, 150, 150],
                                [30, 30, 30],
                                [190, 30, 30]
                                ]).to(device)
        z_y_mu, z_y_logvar = vae_rgb_enc(example)
        y_mu_z_y = vae_rgb_dec(z_y_mu)
        print("encoded colors: {}".format(example))
        print("decoded colors: {}\n".format(y_mu_z_y * 255.0))

        example = torch.tensor([[528, 13],
                                [528, 0],
                                [528, 7]]
                                )
        z_x_mu, z_x_logvar = vae_txt_enc(example.to(device), torch.LongTensor([2,2,2]))
        decoded_colors = vae_rgb_dec(z_x_mu)
        sampled_colors = vae_rgb_dec(torch.randn_like(z_x_mu) * torch.exp(0.5 * z_x_logvar) + z_x_mu)
        print("encoded tests: {}".format(example))
        print("decoded colors: {}\n".format(decoded_colors))
        print("sampled colors: {}\n".format(sampled_colors))

        return

    print("=== begin testing ===")

    losses, mean_accuracies, sample_accuracies, cond_accuracies, diverge_rates, best_epochs = [], [], [], [], [], []
    for iter_num in range(1, args.num_iter + 1):
        filename = 'checkpoint_vae_{}_{}_alpha={}_beta={}_best'.format(args.sup_lvl,
                                                                        iter_num,
                                                                        args.alpha,
                                                                        args.beta)
        
        epoch, train_args, vae_emb, vae_img_enc, vae_txt_enc, vae_mult_enc, vae_img_dec, vae_txt_dec, vocab, vocab_size, pad_index = \
            load_checkpoint(folder=args.load_dir, filename=filename)
        
        vae_emb.to(device)
        vae_img_enc.to(device)
        vae_txt_enc.to(device)
        vae_mult_enc.to(device)
        vae_img_dec.to(device)
        vae_txt_dec.to(device)

        # sanity check
        print("iteration {} with alpha {} and beta {}\n".format(iter_num, train_args.alpha, train_args.beta))
        print("best training epoch: {}".format(epoch))
        # sanity_check()

        test_dataset = Chairs_ReferenceGame(vocab=vocab, split='Test', context_condition=args.context_condition)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=100)

        # compute test loss & reference game accuracy
        # losses.append(test_loss())
        mean_acc, sample_acc, cond_acc, diverge_rate = test_refgame_accuracy()
        
        diverge_rates.append(diverge_rate)
        mean_accuracies.append(mean_acc)
        sample_accuracies.append(sample_acc)
        cond_accuracies.append(cond_acc)
        best_epochs.append(epoch)

    # losses = np.array(losses)
    mean_accuracies = np.array(mean_accuracies)
    sample_accuracies = np.array(sample_accuracies)
    cond_accuracies = np.array(cond_accuracies)
    diverge_rates = np.array(diverge_rates)

    # save files as np arrays
    print("saving file to {} ...".format(args.out_dir))
    np.save(os.path.join(args.out_dir, 'sample_accuracies_{}_alpha={}_beta={}.npy'.format(args.sup_lvl, args.alpha, args.beta)), sample_accuracies)
    np.save(os.path.join(args.out_dir, 'mean_accuracies_{}_alpha={}_beta={}.npy'.format(args.sup_lvl, args.alpha, args.beta)), mean_accuracies)
    np.save(os.path.join(args.out_dir, 'cond_accuracies_{}_alpha={}_beta={}.npy'.format(args.sup_lvl, args.alpha, args.beta)), cond_accuracies)
    np.save(os.path.join(args.out_dir, 'divergence_rates_{}_alpha={}_beta={}.npy'.format(args.sup_lvl, args.alpha, args.beta)), diverge_rates)
    print("... saving complete.")

    print("\n======> Best epochs: {}".format(best_epochs))
    # print("\n======> Average loss: {:6f}".format(np.mean(losses)))
    print("======> Average mean-based accuracy: {:4f}".format(np.mean(mean_accuracies)))

