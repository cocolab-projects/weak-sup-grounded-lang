from __future__ import print_function

import os
import sys
import random
from itertools import chain
import numpy as np
from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from color_dataset import (ColorDataset, WeakSup_ColorDataset)

from utils import (AverageMeter, save_checkpoint, _reparameterize, loss_multimodal, loss_text_unimodal, loss_image_unimodal)
from models import (TextEmbedding, TextEncoder, TextDecoder,
                    ColorEncoder, ColorEncoder_Augmented, MultimodalEncoder, ColorDecoder)
from forward import (forward_vae_rgb_text, forward_vae_rgb, forward_vae_text)

SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

if __name__ == '__main__':
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str, help='where to save checkpoints')
    parser.add_argument('sup_lvl', type=float, default = 1.0,
                        help='how much of the data to supervise [default: 1.0]')
    parser.add_argument('--z_dim', type=int, default=100,
                        help='number of latent dimension [default = 100]')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='word dropout in text generation [default = 0.]')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size [default=100]')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='learning rate [default=0.0003]')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs [default: 50]')
    parser.add_argument('--alpha', type=float, default=1,
                        help='lambda hyperparameter for text loss')
    parser.add_argument('--beta', type=float, default=1,
                        help='lambda hyperparameter for image loss')
    parser.add_argument('--gamma', type=float, default=1,
                        help='lambda hyperparameter for D_KL term loss')
    parser.add_argument('--weaksup', action='store_true',
                        help='whether unpaired datasets are trained')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_iter', type=int, default = 1,
                        help='number of iterations for this setting [default: 1]')
    parser.add_argument('--context_condition', type=str, default='far', help='whether the dataset is to include all data')
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)

    def train(epoch):
        """Function: train
        Args:
            param1 (int) epoch: training epoch
        Returns:
            (float): training loss over epoch
        """
        vae_emb.train()
        vae_rgb_enc.train()
        vae_txt_enc.train()
        vae_mult_enc.train()
        vae_rgb_dec.train()
        vae_txt_dec.train()

        loss_meter = AverageMeter()
        pbar = tqdm(total=len(train_xy_loader))
        for batch_idx, (y_rgb, x_tgt, x_src, x_len) in enumerate(train_xy_loader):
            batch_size = y_rgb.size(0)

            models_xy = (vae_txt_enc, vae_rgb_enc, vae_mult_enc, vae_txt_dec, vae_rgb_dec)
            out = forward_vae_rgb_text((y_rgb, x_tgt, x_src, x_len), models_multimodal)
            out['pad_index'] = pad_index

            # compute loss
            loss = loss_multimodal(out, batch_size, alpha=args.alpha, beta=args.beta)

            # update based on loss
            loss_meter.update(loss.item(), batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': loss_meter.avg})
            pbar.update()
        pbar.close()
        
        if epoch % 10 == 0:
            print('====> Train Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
        
        return loss_meter.avg

    def train_weakly_supervised(epoch):
        """Function: train
        Args:
            param1 (int) epoch: training epoch
        Returns:
            (float): training loss over epoch
        """
        vae_emb.train()
        vae_rgb_enc.train()
        vae_txt_enc.train()
        vae_mult_enc.train()
        vae_rgb_dec.train()
        vae_txt_dec.train()

        train_xy_iterator = train_xy_loader.__iter__()
        train_x_iterator = train_x_loader.__iter__()
        train_y_iterator = train_y_loader.__iter__()

        models_xy = (vae_txt_enc, vae_rgb_enc, vae_mult_enc, vae_txt_dec, vae_rgb_dec)
        models_x = (vae_txt_enc, vae_txt_dec)
        models_y = (vae_rgb_enc, vae_rgb_dec)
        
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(train_y_iterator))
        for batch_i in range(len(train_y_iterator)):
            try:
                data_xy_args = list(next(train_xy_iterator))
            except StopIteration:
                train_xy_iterator = train_xy_loader.__iter__()
                data_xy_args = list(next(train_xy_iterator))

            try:
                _, x_tgt, x_src, x_len = next(train_x_iterator)
                data_x_args = [x_tgt, x_src, x_len]
            except StopIteration:
                train_x_iterator = train_x_loader.__iter__()
                _, x_tgt, x_src, x_len = next(train_x_iterator)
                data_x_args = [x_tgt, x_src, x_len]

            try:
                y, _, _, _ = next(train_y_iterator)
                data_y_args = [y]
            except StopIteration:
                train_y_iterator = train_y_loader.__iter__()
                y, _, _, _ = next(train_y_iterator)
                data_y_args = [y]

            batch_size = min(data_x_args[0].size(0), data_y_args[0].size(0), data_xy_args[0].size(0))
            
            # cast elements to CUDA and keep only the batch_size so that sizes are all the same
            for j in range(len(data_xy_args)):
                data_xy_args[j] = data_xy_args[j][:batch_size]
                data_xy_args[j] = data_xy_args[j].to(device)
            
            for j in range(len(data_x_args)):
                data_x_args[j] = data_x_args[j][:batch_size]
                data_x_args[j] = data_x_args[j].to(device)
            
            for j in range(len(data_y_args)):
                data_y_args[j] = data_y_args[j][:batch_size]
                data_y_args[j] = data_y_args[j].to(device)

            output_xy_dict = forward_vae_rgb_text(data_xy_args, models_xy)
            output_x_dict = forward_vae_text(data_x_args, models_x)
            output_y_dict = forward_vae_rgb(data_y_args, models_y)
            output_xy_dict['pad_index'] = pad_index
            output_x_dict['pad_index'] = pad_index

            loss_xy = loss_multimodal(output_xy_dict, batch_size, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
            loss_x = loss_text_unimodal(output_x_dict, batch_size, alpha=args.alpha, gamma=args.gamma)
            loss_y = loss_image_unimodal(output_y_dict, batch_size, beta=args.beta, gamma=args.gamma)

            loss = loss_xy + loss_x + loss_y

            loss_meter.update(loss.item(), batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': loss_meter.avg})
            pbar.update()
        pbar.close()

        if epoch % 10 == 0:
            print('====> Train Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
        
        return loss_meter.avg


    def test(epoch):
        vae_emb.eval()
        vae_rgb_enc.eval()
        vae_txt_enc.eval()
        vae_mult_enc.eval()
        vae_rgb_dec.eval()
        vae_txt_dec.eval()

        with torch.no_grad():
            loss_meter = AverageMeter()
            pbar = tqdm(total=len(test_loader))

            models_xy = (vae_txt_enc, vae_rgb_enc, vae_mult_enc, vae_txt_dec, vae_rgb_dec)
            models_x = (vae_txt_enc, vae_txt_dec)
            models_y = (vae_rgb_enc, vae_rgb_dec)

            for batch_idx, (y_rgb, x_tgt, x_src, x_len) in enumerate(test_loader):
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
                loss = loss_multimodal(out, batch_size, alpha=args.alpha, beta=args.beta)
                loss_meter.update(loss.item(), batch_size)

                pbar.update()
            pbar.close()
            if epoch % 10 == 0:
                print('====> Test Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
        return loss_meter.avg

    print("=== begin training ===")
    print("args: sup_lvl: {} alpha: {} beta: {} seed: {} context condition?: {} cuda?: {} weaksup: {}".format(args.sup_lvl,
                                                                                    args.alpha,
                                                                                    args.beta,
                                                                                    args.seed,
                                                                                    args.context_condition,
                                                                                    args.cuda,
                                                                                    args.weaksup))

    # repeat training on same model w/ different random seeds for |num_iter| times
    for i in range(1, args.num_iter + 1):
        print("\nTraining iteration {} for supervision level {}".format(i, args.sup_lvl))
        
        # set random seeds
        random_iter_seed = random.randint(0, 500)
        print("Random seed set to : {}".format(random_iter_seed))
        torch.cuda.manual_seed(random_iter_seed)
        random.seed(random_iter_seed)
        torch.manual_seed(random_iter_seed)
        np.random.seed(random_iter_seed)

        # set learning device
        args.cuda = args.cuda and torch.cuda.is_available()
        device = torch.device('cuda' if args.cuda else 'cpu')

        # Define training dataset & build vocab
        train_dataset = WeakSup_ColorDataset(supervision_level=args.sup_lvl, context_condition=args.context_condition)
        train_xy_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        N_mini_batches = len(train_xy_loader)
        vocab_size = train_dataset.vocab_size
        vocab = train_dataset.vocab
        w2i = vocab['w2i']
        pad_index = w2i[PAD_TOKEN]

        # Define test dataset
        test_dataset = ColorDataset(vocab=vocab, split='Validation', context_condition=args.context_condition)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

        if args.weaksup:
            unpaired_dataset = ColorDataset(vocab=vocab, split='Train', context_condition=args.context_condition)
            train_x_loader = DataLoader(unpaired_dataset, shuffle=True, batch_size=args.batch_size)
            train_y_loader = DataLoader(unpaired_dataset, shuffle=True, batch_size=args.batch_size)

        # Define latent dimension |z_dim|
        z_dim = args.z_dim

        # Define model
        vae_emb = TextEmbedding(vocab_size)
        vae_rgb_enc = ColorEncoder(z_dim)
        vae_txt_enc = TextEncoder(vae_emb, z_dim)
        vae_mult_enc = MultimodalEncoder(vae_emb, z_dim)
        vae_rgb_dec = ColorDecoder(z_dim)
        vae_txt_dec = TextDecoder(vae_emb, z_dim, w2i[SOS_TOKEN], w2i[EOS_TOKEN],
                                    w2i[PAD_TOKEN], w2i[UNK_TOKEN], word_dropout=args.dropout)

        # Mount devices unto GPU
        vae_emb = vae_emb.to(device)
        vae_rgb_enc = vae_rgb_enc.to(device)
        vae_txt_enc = vae_txt_enc.to(device)
        vae_mult_enc = vae_mult_enc.to(device)
        vae_rgb_dec = vae_rgb_dec.to(device)
        vae_txt_dec = vae_txt_dec.to(device)

        # Define optimizer
        optimizer = torch.optim.Adam(
            chain(
            vae_emb.parameters(),
            vae_rgb_enc.parameters(),
            vae_txt_enc.parameters(),
            vae_mult_enc.parameters(),
            vae_rgb_dec.parameters(),
            vae_txt_dec.parameters(),
        ), lr=args.lr)

        best_loss = float('inf')
        track_loss = np.zeros((args.epochs, 2))
        
        for epoch in range(1, args.epochs + 1):
            train_loss = train(epoch) if not args.weaksup else train_weakly_supervised(epoch)
            test_loss = test(epoch)

            is_best = test_loss < best_loss
            best_loss = min(test_loss, best_loss)
            track_loss[epoch - 1, 0] = train_loss
            track_loss[epoch - 1, 1] = test_loss
            
            save_checkpoint({
                'epoch': epoch,
                'vae_emb': vae_emb.state_dict(),
                'vae_rgb_enc': vae_rgb_enc.state_dict(),
                'vae_txt_enc': vae_txt_enc.state_dict(),
                'vae_mult_enc': vae_mult_enc.state_dict(),
                'vae_rgb_dec': vae_rgb_dec.state_dict(),
                'vae_txt_dec': vae_txt_dec.state_dict(),
                'optimizer': optimizer.state_dict(),
                'track_loss': track_loss,
                'cmd_line_args': args,
                'vocab': vocab,
                'vocab_size': vocab_size,
                'seed': random_iter_seed
            }, is_best, folder=args.out_dir,
            filename='checkpoint_vae_{}_{}_alpha={}_beta={}'.format(args.sup_lvl, i, args.alpha, args.beta))
            np.save(os.path.join(args.out_dir,
                'loss_{}_{}.npy'.format(args.sup_lvl, i)), track_loss)


