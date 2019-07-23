import os
import sys
import numpy as np
import collections
import pickle
from tqdm import tqdm

import torch 
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal

from utils import (AverageMeter, score_txt_logits, _reparameterize,
                    loss_multimodal, _log_mean_exp, gaussian_log_pdf, isotropic_gaussian_log_pdf,
                    bernoulli_log_pdf, get_text, get_image_text_joint_nll)
from models import (TextEmbedding, TextEncoder, TextDecoder,
                    ColorEncoder, ColorEncoder_Augmented, MultimodalEncoder, ColorDecoder)
from color_dataset import (ColorDataset, Colors_ReferenceGame)

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

        test_dataset = ColorDataset(vocab=vocab, split=split, context_condition=args.context_condition)
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
                z_sample_x = _reparameterize(z_x_mu, z_x_logvar)
                z_sample_y = _reparameterize(z_y_mu, z_y_logvar)
                z_sample_xy = _reparameterize(z_xy_mu, z_xy_logvar)

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
                        'y': y_rgb, 'x': x_tgt, 'x_len': x_len, 'pad_index': pad_index}

                # compute loss
                loss = loss_multimodal(out, batch_size, alpha=train_args.alpha, beta=train_args.beta)
                loss_meter.update(loss.item(), batch_size)

            print('====> Final Test Loss: {:.4f}'.format(loss_meter.avg))
        return loss_meter.avg

    def get_sampled_joint_prob(y_i, x_src, x_tgt, x_len, z_xy_mu, z_xy_logvar):
        z_samples = torch.randn(N_SAMPLE, train_args.z_dim).to(device) * torch.exp(0.5 * z_xy_logvar) + z_xy_mu

        y_mu_list = vae_rgb_dec(z_samples)
        x_tgt_logits_list = vae_txt_dec(z_samples, x_src.unsqueeze(0).repeat(N_SAMPLE, 1),
                                                    x_len.unsqueeze(0).repeat(N_SAMPLE))
        elt_max_len = x_tgt_logits_list.size(1)
        x_tgt_i = x_tgt[:elt_max_len]
        x_len_i = elt_max_len

        return get_image_text_joint_nll(y_i, y_mu_list, x_tgt_i, x_tgt_logits_list, x_len, z_samples, z_xy_mu, z_xy_logvar, pad_index)

    def get_mean_joint_prob(y_rgb, x_src, x_tgt, x_len, z_xy_mu, z_xy_logvar, verbose=False):
        y_mu_z_xy = vae_rgb_dec(z_xy_mu.unsqueeze(0))
        x_tgt_logits = vae_txt_dec(z_xy_mu.unsqueeze(0), x_src.unsqueeze(0), x_len.unsqueeze(0))
        x_tgt = x_tgt[:x_len]

        return get_image_text_joint_nll(y_rgb, y_mu_z_xy, x_tgt, x_tgt_logits, x_len, z_xy_mu.unsqueeze(0), z_xy_mu, z_xy_logvar, pad_index, verbose)

    def get_conditional_choice(y_1, y_2, y_3, z_mu):
        pred_rgb_cond = vae_rgb_dec(z_mu)
        diff_tgt = torch.mean(torch.pow(pred_rgb_cond - y_1, 2))
        diff_d1 = torch.mean(torch.pow(pred_rgb_cond - y_2, 2))
        diff_d2 = torch.mean(torch.pow(pred_rgb_cond - y_3, 2))
        return pred_rgb_cond, 1 if (diff_tgt < diff_d1 and diff_tgt < diff_d2) else (2 if diff_d1 < diff_d2 else 3)

    def test_refgame_accuracy():
        """Function: test_refgame_accuracy
        Returns:
            (float) mean_acc: mean-based choice accuracy
            (float) sample_acc: sample-based choice accuracy
            (float) diverge_rate: choice divergence between mean-based and sample-based choice
        Compute final accuracy test on reference game dataset
        """
        print("Computing final accuracy for reference game settings...")

        ref_dataset = Colors_ReferenceGame(vocab, split='Test', context_condition=args.context_condition)
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
            total_count_dict = collections.defaultdict(int)
            mean_correct_count_dict = collections.defaultdict(int)
            mean_correct_count, sample_correct_count, cond_correct_count, diverge_count = 0, 0, 0, 0
            pz_correct_count = 0
            pz_dict = collections.defaultdict(int)

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
                    z_y1_mu, z_y1_logvar = vae_rgb_enc(y_rgb)
                    z_y2_mu, z_y2_logvar = vae_rgb_enc(d1_rgb)
                    z_y3_mu, z_y3_logvar = vae_rgb_enc(d2_rgb)
                    z_xy_mu_tgt, z_xy_logvar_tgt = vae_mult_enc(y_rgb, x_src, x_len)
                    z_xy_mu_d1, z_xy_logvar_d1 = vae_mult_enc(d1_rgb, x_src, x_len)
                    z_xy_mu_d2, z_xy_logvar_d2 = vae_mult_enc(d2_rgb, x_src, x_len)

                    # check accuracy for each datapoint via mean, sampling, conditional
                    for i in range(batch_size):
                        verbose = i % 100 == 0
                        # verbose = False
                        total_count += 1
                        total_count_dict[x_len[i].item()] += 1

                        # # sample-based estimator of joint probabilities based on z ~ q(z|x,y)
                        # p_x_y1_sampled = get_sampled_joint_prob(y_rgb[i], x_src[i], x_tgt[i], x_len[i], z_xy_mu_tgt[i], z_xy_logvar_tgt[i])
                        # p_x_y2_sampled = get_sampled_joint_prob(d1_rgb[i], x_src[i], x_tgt[i], x_len[i], z_xy_mu_d1[i], z_xy_logvar_d1[i])
                        # p_x_y3_sampled = get_sampled_joint_prob(d2_rgb[i], x_src[i], x_tgt[i], x_len[i], z_xy_mu_d2[i], z_xy_logvar_d2[i])

                        if verbose: print("\n========= NEW IMAGE TEXT DATAPOINT ==========")
                        # mean-based estimator of joint probabilities based on z ~ q(z|x,y)
                        p_x_y1_mean = get_mean_joint_prob(y_rgb[i], x_src[i], x_tgt[i], x_len[i], z_xy_mu_tgt[i], z_xy_logvar_tgt[i], verbose=verbose)
                        p_x_y2_mean = get_mean_joint_prob(d1_rgb[i], x_src[i], x_tgt[i], x_len[i], z_xy_mu_d1[i], z_xy_logvar_d1[i], verbose=verbose)
                        p_x_y3_mean = get_mean_joint_prob(d2_rgb[i], x_src[i], x_tgt[i], x_len[i], z_xy_mu_d2[i], z_xy_logvar_d2[i], verbose=verbose)

                        # p(z)-based estimator of joint probabilities based on z ~ q(z|x,y)
                        p_z_x_y1 = torch.sum(isotropic_gaussian_log_pdf(z_xy_mu_tgt[i].unsqueeze(0)), dim=1)
                        p_z_x_y2 = torch.sum(isotropic_gaussian_log_pdf(z_xy_mu_d1[i].unsqueeze(0)), dim=1)
                        p_z_x_y3 = torch.sum(isotropic_gaussian_log_pdf(z_xy_mu_d2[i].unsqueeze(0)), dim=1)

                        # choice based on conditional distribution z ~ q(z|x)
                        pred_rgb_cond, cond_choice = get_conditional_choice(y_rgb[i], d1_rgb[i], d2_rgb[i], z_x_mu[i])
                        
                        mean_correct, sample_correct, cond_correct, diverge = False, False, False, False
                        pz_correct = False
                        pz_dict[np.argmin(np.array([p_z_x_y1, p_z_x_y2, p_z_x_y3]))] += 1

                        mean_choice = np.argmin(np.array([p_x_y1_mean, p_x_y2_mean, p_x_y3_mean]))
                        pz_choice = np.argmin(np.array([p_z_x_y1, p_z_x_y2, p_z_x_y3]))
                        if pz_choice == 0:
                            pz_correct_count += 1
                            pz_correct = True
                        if mean_choice == 0:
                            mean_correct_count += 1
                            mean_correct = True
                            mean_correct_count_dict[x_len[i].item()] += 1
                        # if p_x_y1_sampled < p_x_y2_sampled and p_x_y1_sampled < p_x_y3_sampled:
                        #     sample_correct_count += 1
                        #     sample_correct = True
                        if cond_choice == 1:
                            cond_correct_count += 1
                            cond_correct = True
                        if pz_choice == 0 and mean_choice != 0:
                            diverge_count += 1
                            diverge = True
                        if verbose:
                            match_text = get_text(vocab['i2w'], x_tgt[i], x_len[i])
                            print("color: {} <==> text: {} == {}".format(y_rgb[i], match_text, x_tgt[i][:x_len[i]]))
                            print("predicted rgb based on conditional: {}".format(pred_rgb_cond))
                            print("mean-based choice correct? {}".format('T' if mean_correct else 'F'))
                            # print("sample-based choice correct? {}".format('T' if sample_correct else 'F'))
                            # print("conditional distribution-based choice correct? {}".format('T' if cond_correct else 'F'))
                            # print("p_x_y1_sampled: {}, p_x_y2_sampled: {}, p_x_y3_sampled: {}".format(p_x_y1_sampled, p_x_y2_sampled, p_x_y3_sampled))
                            print("p_x_y1_mean: {}, p_x_y2_mean: {}, p_x_y3_mean: {}".format(p_x_y1_mean, p_x_y2_mean, p_x_y3_mean))
                            print("p_z_x_y1: {}, p_z_x_y2: {}, p_z_x_y3: {}".format(p_z_x_y1, p_z_x_y2, p_z_x_y3))
                            
                            print("\npz argmax info: {}".format(pz_dict))
                            print("current mean-based choice accuracy: {}".format(mean_correct_count / total_count))
                            print("current pz-based choice accuracy: {}".format(pz_correct_count / total_count))
                            print("current diverge rate: {}".format(diverge_count / total_count))
                            print("current mean histogram: {}".format([(i, total_count_dict[i], mean_correct_count_dict[i] / total_count_dict[i]) \
                                                                                    for i in range(2, 11) if total_count_dict[i] != 0]))
                            print("================END OF IMAGE TEXT DATAPOINT ====================")
                    pbar.update()

            pz_acc = pz_correct_count / float(total_count) * 100
            mean_acc = mean_correct_count / float(total_count) * 100
            # sample_acc = sample_correct_count / float(total_count) * 100
            cond_acc = cond_correct_count / float(total_count) * 100
            diverge_rate = diverge_count / float(total_count) * 100
            print('====> Final p(z)-based Accuracy: {}/{} = {}%'.format(pz_correct_count, total_count, pz_acc))
            print('====> Final Mean-based Accuracy: {}/{} = {}%'.format(mean_correct_count, total_count, mean_acc))
            print('====> Final Conditional Accuracy: {}/{} = {}%'.format(cond_correct_count, total_count, cond_acc))
            print('====> Final Divergence Rate: {}/{} = {}%\n'.format(diverge_count, total_count, diverge_rate))
            print('----> total_count_dict: {}'.format(total_count_dict))
            print('----> mean_correct_count_dict: {}'.format(mean_correct_count_dict))
        return mean_acc, pz_acc, cond_acc, pz_dict, mean_correct_count_dict, total_count_dict

    def load_checkpoint(folder='./', filename='model_best'):
        print("\nloading checkpoint file: {}.pth.tar ...\n".format(filename)) 
        checkpoint = torch.load(os.path.join(folder, filename + '.pth.tar'))
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

    losses, mean_accuracies, pz_accuracies, cond_accuracies, pz_dicts, best_epochs = [], [], [], [], [], []
    for iter_num in range(1, args.num_iter + 1):

        filename = 'checkpoint_vae_{}_{}_alpha={}_beta={}_best'.format(args.sup_lvl,
                                                                        iter_num,
                                                                        args.alpha,
                                                                        args.beta)
        
        epoch, train_args, vae_emb, vae_rgb_enc, vae_txt_enc, vae_mult_enc, vae_rgb_dec, vae_txt_dec, vocab, vocab_size, pad_index = \
            load_checkpoint(folder=args.load_dir, filename=filename)
        
        vae_emb.to(device)
        vae_rgb_enc.to(device)
        vae_txt_enc.to(device)
        vae_mult_enc.to(device)
        vae_rgb_dec.to(device)
        vae_txt_dec.to(device)

        # sanity check
        print("iteration {} with alpha {} and beta {}\n".format(iter_num, train_args.alpha, train_args.beta))
        print("best training epoch: {}".format(epoch))
        # sanity_check()

        # compute test loss & reference game accuracy
        
        mean_acc, pz_acc, cond_acc, pz_dict, mean_correct_count_dict, total_count_dict = test_refgame_accuracy()
        mean_accuracies.append(mean_acc)
        pz_accuracies.append(pz_acc)
        cond_accuracies.append(cond_acc)
        best_epochs.append(epoch)

    # losses = np.array(losses)
    mean_accuracies = np.array(mean_accuracies)
    pz_accuracies = np.array(pz_accuracies)
    cond_accuracies = np.array(cond_accuracies)

    accuracy_by_len_dict = {i:(total_count_dict[i], mean_correct_count_dict[i]) for i in range(2, 11) if total_count_dict[i] != 0}
    # save files as np arrays
    print("\nsaving file to {} ...".format(args.out_dir))
    f_pz = open(os.path.join(args.out_dir, 'pz_dict_{}_alpha={}_beta={}.npy'.format(args.sup_lvl, args.alpha, args.beta)), "wb")
    pickle.dump(pz_dict, f_pz)
    f_pz.close()

    f_bylen = open(os.path.join(args.out_dir, 'mean_correct_count_dict_{}_alpha={}_beta={}.npy'.format(args.sup_lvl, args.alpha, args.beta)), "wb")
    pickle.dump(accuracy_by_len_dict, f_bylen)
    f_bylen.close()

    np.save(os.path.join(args.out_dir, 'pz_accs_{}_alpha={}_beta={}.npy'.format(args.sup_lvl, args.alpha, args.beta)), pz_accuracies)
    np.save(os.path.join(args.out_dir, 'mean_accuracies_{}_alpha={}_beta={}.npy'.format(args.sup_lvl, args.alpha, args.beta)), mean_accuracies)
    np.save(os.path.join(args.out_dir, 'cond_accuracies_{}_alpha={}_beta={}.npy'.format(args.sup_lvl, args.alpha, args.beta)), cond_accuracies)
    print("... saving complete.")

    print("\nmean_accuracies: {}".format(mean_accuracies))
    print("pz_accuracies: {}".format(pz_accuracies))
    print("histogram: {} ".format([(i, total_count_dict[i], mean_correct_count_dict[i] / total_count_dict[i]) for i in range(2, 11) if total_count_dict[i] != 0]))
    print("pz_dict: {}".format(pz_dict))

    print("\n======> Best epochs: {}".format(best_epochs))
    print("======> Average mean-based accuracy: {:4f}".format(np.mean(mean_accuracies)))

