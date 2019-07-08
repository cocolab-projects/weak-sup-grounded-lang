import os
import sys
import numpy as np

import torch 
from torch.utils.data import DataLoader
from utils import (AverageMeter)
from models import (TextEmbedding, TextEncoder, TextDecoder,
                    ColorEncoder, MultimodalEncoder, ColorDecoder)
from color_dataset import (ColorDataset, Colors_ReferenceGame)

SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

if __name__ == '__main__':
    def test_loss(vocab, split='Test'):
        '''
        Test model on newly seen dataset -- gives final test loss
        '''
        assert vocab != None
        print("Computing final test loss on newly seen dataset...")

        test_dataset = ColorDataset(vocab=vocab, split=split, hard=args.hard)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=100)

        model.eval()
        with torch.no_grad():
            loss_meter = AverageMeter()

            for batch_idx, (y_rgb, x_inp, x_len) in enumerate(test_loader):
                batch_size = x_src.size(0) 
                y_rgb = y_rgb.to(device).float()
                x_src = x_src.to(device)
                x_tgt = x_tgt.to(device)
                x_len = x_len.to(device)

                # Encode to |z|
                z_x_mu, z_x_logvar = vae_txt_enc(x_tgt, x_len)
                z_y_mu, z_y_logvar = vae_rgb_enc(y_rgb)
                z_xy_mu, z_xy_logvar = vae_mult_enc(y_rgb, x_tgt, x_len)

                # sample via reparametrization
                z_sample_x = reparameterize(z_x_mu, z_x_logvar)
                z_sample_y = reparameterize(z_y_mu, z_y_logvar)
                z_sample_xy = reparameterize(z_xy_mu, z_xy_logvar)

                # "predictions"
                y_mu_z_y = vae_rgb_dec(z_sample_y)
                y_mu_z_xy = vae_rgb_dec(z_sample_xy)
                x_logit_z_x = vae_txt_dec(z_sample_x, x_tgt, x_len)
                x_logit_z_xy = vae_txt_dec(z_sample_xy, x_tgt, x_len)

                out = {'z_x_mu': z_x_mu, 'z_x_logvar': z_x_logvar,
                        'z_y_mu': z_y_mu, 'z_y_logvar': z_y_logvar,
                        'z_xy_mu': z_xy_mu, 'z_xy_logvar': z_xy_logvar,
                        'y_mu_z_y': y_mu_z_y, 'y_mu_z_xy': y_mu_z_xy, 
                        'x_logit_z_x': x_logit_z_x, 'x_logit_z_xy': x_logit_z_xy,
                        'y': y_rgb, 'x_tgt': x_tgt}

                # compute loss
                loss = loss_multimodal(out, batch_size)
                loss_meter.update(loss.item(), batch_size)

                pbar.update()
            pbar.close()
            print('====> Final Test Loss: {:.4f}'.format(loss_meter.avg))
        return loss_meter.avg

    def test_refgame_accuracy(model, vocab):
        '''
        Final accuracy: test on reference game dataset
        '''
        assert vocab != None
        print("Computing final accuracy for reference game settings...")

        ref_dataset = Colors_ReferenceGame(vocab, split='Test', hard=args.hard)
        ref_loader = DataLoader(ref_dataset, shuffle=False, batch_size=100)

        model.eval()

        with torch.no_grad():
            loss_meter = AverageMeter()

            total_count = 0
            correct_count = 0

            for batch_idx, (tgt_rgb, d1_rgb, d2_rgb, x_inp, x_len) in enumerate(ref_loader):
                batch_size = x_src.size(0) 
                y_rgb = y_rgb.to(device).float()
                x_src = x_src.to(device)
                x_tgt = x_tgt.to(device)
                x_len = x_len.to(device)

                # Encode to |z|
                z_x_mu, z_x_logvar = vae_txt_enc(x_tgt, x_len)
                z_y_mu, z_y_logvar = vae_rgb_enc(y_rgb)
                z_xy_mu, z_xy_logvar = vae_mult_enc(y_rgb, x_tgt, x_len)

                # sample via reparametrization
                z_sample_x = reparameterize(z_x_mu, z_x_logvar)
                z_sample_y = reparameterize(z_y_mu, z_y_logvar)
                z_sample_xy = reparameterize(z_xy_mu, z_xy_logvar)

                # "predictions"
                y_mu_z_y = vae_rgb_dec(z_sample_y)
                y_mu_z_xy = vae_rgb_dec(z_sample_xy)

                # compute loss
                loss = loss_multimodal(out, batch_size)
                loss_meter.update(loss.item(), batch_size)

                for i in range(batch_size):
                    diff_tgt = torch.mean(torch.pow(pred_rgb[i] - tgt_rgb[i], 2))
                    diff_d1 = torch.mean(torch.pow(pred_rgb[i] - d1_rgb[i], 2))
                    diff_d2 = torch.mean(torch.pow(pred_rgb[i] - d2_rgb[i], 2))
                    total_count += 1
                    if diff_tgt.item() < diff_d1.item() and diff_tgt.item() < diff_d2.item():
                        correct_count += 1

                loss_meter.update(loss.item(), batch_size)

            accuracy = correct_count / float(total_count) * 100
            print('====> Final Test Loss: {:.4f}'.format(loss_meter.avg))
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
        w2i = vocab['w2i']
        pad_index = w2i[PAD_TOKEN]
        vocab_size = checkpoint['vocab_size']
        args = checkpoint['cmd_line_args']

        vae_emb = TextEmbedding(vocab_size)
        vae_rgb_enc = ColorEncoder(args.z_dim)
        vae_txt_enc = TextEncoder(vae_emb, args.z_dim)
        vae_mult_enc = MultimodalEncoder(vae_emb, args.z_dim)
        vae_rgb_dec = ColorDecoder(args.z_dim)
        vae_txt_dec = TextDecoder(vae_emb, args.z_dim, w2i[SOS_TOKEN], w2i[EOS_TOKEN],
                                    w2i[PAD_TOKEN], w2i[UNK_TOKEN], word_dropout=args.word_dropout)
        vae_emb.load_state_dict(vae_emb_sd)
        vae_rgb_enc.load_state_dict(vae_rgb_enc_sd)
        vae_txt_enc.load_state_dict(vae_txt_enc_sd)
        vae_mult_enc.load_state_dict(vae_mult_enc_sd)
        vae_rgb_dec.load_state_dict(vae_rgb_dec_sd)
        vae_txt_dec.load_state_dict(vae_txt_dec_sd)

        return epoch, args, vae_emb, vae_rgb_enc, vae_txt_enc, vae_mult_enc, vae_rgb_dec, vae_txt_dec, vocab, vocab_size

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir', type=str, help='where to load checkpoints from')
    parser.add_argument('out_dir', type=str, help='where to store results from')
    parser.add_argument('--sup_lvl', type=float, help='supervision level, if any')
    parser.add_argument('--num_iter', type=int, default=1,
                        help='number of total iterations performed on each setting [default: 1]')
    parser.add_argument('--hard', action='store_true', help='whether the dataset is to be easy')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    losses, accuracies = [], []
    for iter_num in range(1, args.num_iter + 1):
        epoch, vae_emb, vae_rgb_enc, vae_txt_enc, vae_mult_enc, vae_rgb_dec, vae_txt_dec, vocab, vocab_size = \
            load_checkpoint(folder=args.load_dir,
                            filename='checkpoint_{}_{}_best'.format(args.sup_lvl, iter_num))
        print("iteration {}".format(iter_num))
        print("best training epoch: {}".format(epoch))

        losses.append(test_loss(vocab))
        accuracies.append(test_refgame_accuracy(vocab))

    losses = np.array(losses)
    accuracies = np.array(accuracies)
    np.save(os.path.join(args.out_dir, 'accuracies_{}.npy'.format(args.sup_lvl)), accuracies)
    np.save(os.path.join(args.out_dir, 'final_losses_{}.npy'.format(args.sup_lvl)), losses)

    print()
    print("======> Average loss: {}".format(np.mean(losses)))
    print("======> Average accuracy: {}".format(np.mean(accuracies)))
