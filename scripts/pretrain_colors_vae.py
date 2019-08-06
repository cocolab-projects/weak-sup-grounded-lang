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

from utils import (AverageMeter, save_checkpoint, _reparameterize, loss_multimodal, loss_text_unimodal, loss_image_unimodal, loss_multimodal_only)
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
    parser.add_argument('--beta', type=float, default=10,
                        help='lambda hyperparameter for image loss')
    parser.add_argument('--gamma', type=float, default=1,
                        help='lambda hyperparameter for D_KL term loss')
    parser.add_argument('--weaksup', type=str, default='default',
                        help='mode for unpaired dataset training')
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

    def train_rgb_only(epoch, optimizer):
        """Function: train
        Args:
            param1 (int) epoch: training epoch
        Returns:
            (float): training loss over epoch
        """
        vae_rgb_enc.train()
        vae_rgb_dec.train()

        models_y = (vae_rgb_enc, vae_rgb_dec)
        
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(train_y_loader))
        for batch_idx, (y_rgb, _, _, _) in enumerate(train_y_loader):
            batch_size = y_rgb.size(0)
            y_rgb = y_rgb.to(device).float()
            data_y_args = [y_rgb]

            output_y_dict = forward_vae_rgb(data_y_args, models_y)
            loss_y = loss_image_unimodal(output_y_dict, batch_size, beta=args.beta, gamma=args.gamma)

            loss_meter.update(loss_y.item(), batch_size)
            optimizer.zero_grad()
            loss_y.backward()
            optimizer.step()

            pbar.set_postfix({'rgb_loss': loss_meter.avg})
            pbar.update()
        pbar.close()

        if epoch % 10 == 0:
            print('====> RGB Train Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
        
        return loss_meter.avg

    def train_text_only(epoch, optimizer):
        vae_emb.train()
        vae_txt_enc.train()
        vae_txt_dec.train()

        models_x = (vae_txt_enc, vae_txt_dec)
        
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(train_x_loader))
        for batch_idx, (_, x_src, x_tgt, x_len) in enumerate(train_x_loader):
            batch_size = x_src.size(0)
            x_tgt = x_tgt.to(device)
            x_src = x_src.to(device)
            x_len = x_len.to(device)
            data_x_args = [x_src, x_tgt, x_len]

            output_x_dict = forward_vae_text(data_x_args, models_x)
            output_x_dict['pad_index'] = pad_index

            loss_x = loss_text_unimodal(output_x_dict, batch_size, alpha=args.alpha, gamma=args.gamma)

            loss_meter.update(loss_x.item(), batch_size)
            optimizer.zero_grad()
            loss_x.backward()
            optimizer.step()

            pbar.set_postfix({'txt_loss': loss_meter.avg})
            pbar.update()
        pbar.close()

        if epoch % 10 == 0:
            print('====> Text Train Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
        
        return loss_meter.avg

    def train_text_only_rev(epoch, optimizer):
        vae_emb.train()
        vae_txt_enc.train()
        vae_txt_dec.train()

        models_x = (vae_txt_enc, vae_txt_dec)
        
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(train_x_loader))
        for batch_idx, (_, x_src, x_tgt, x_len) in enumerate(train_x_loader):
            batch_size = x_src.size(0)
            x_tgt = x_tgt.to(device)
            x_src = x_src.to(device)
            x_len = x_len.to(device)

            ###### !!!!! INPUT ARGUMENTS (x_src, x_tgt) IN REVERSE ORDER !!!!!
            data_x_args = [x_tgt, x_src, x_len]

            output_x_dict = forward_vae_text(data_x_args, models_x)
            output_x_dict['pad_index'] = pad_index

            loss_x = loss_text_unimodal(output_x_dict, batch_size, alpha=args.alpha, gamma=args.gamma)

            loss_meter.update(loss_x.item(), batch_size)
            optimizer.zero_grad()
            loss_x.backward()
            optimizer.step()

            pbar.set_postfix({'txt_loss': loss_meter.avg})
            pbar.update()
        pbar.close()

        if epoch % 10 == 0:
            print('====> Text Train Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
        
        return loss_meter.avg

    def test(epoch):
        vae_emb.eval()
        vae_txt_enc.eval()
        vae_txt_dec.eval()
        vae_rgb_enc.eval()
        vae_rgb_dec.eval()

        with torch.no_grad():
            text_loss_meter = AverageMeter()
            rgb_loss_meter = AverageMeter()
            pbar = tqdm(total=len(test_loader))

            models_x = (vae_txt_enc, vae_txt_dec)
            models_y = (vae_rgb_enc, vae_rgb_dec)

            for batch_idx, (y_rgb, x_src, x_tgt, x_len) in enumerate(test_loader):
                batch_size = y_rgb.size(0)
                y_rgb = y_rgb.to(device).float()
                x_tgt = x_tgt.to(device)
                x_src = x_src.to(device)
                x_len = x_len.to(device)

                data_x_args = [x_tgt, x_src, x_len] if args.weaksup.endswith('reverse') else [x_src, x_tgt, x_len]
                data_y_args = [y_rgb]

                output_x_dict = forward_vae_text(data_x_args, models_x)
                output_x_dict['pad_index'] = pad_index
                output_y_dict = forward_vae_rgb(data_y_args, models_y)

                loss_x = loss_text_unimodal(output_x_dict, batch_size, alpha=args.alpha, gamma=args.gamma)
                loss_y = loss_image_unimodal(output_y_dict, batch_size, beta=args.beta, gamma=args.gamma)

                text_loss_meter.update(loss_x.item(), batch_size)
                rgb_loss_meter.update(loss_y.item(), batch_size)

                pbar.update()
            pbar.close()
            if epoch % 10 == 0:
                print('====> Test Epoch: {}\tRGB Test Loss: {:.4f} Text Test Loss: {:.4f}'.format(epoch, rgb_loss_meter.avg, text_loss_meter.avg))
        return text_loss_meter.avg, rgb_loss_meter.avg

    print("=== begin pretraining ===")
    print("args: alpha: {} beta: {} seed: {} context condition?: {} cuda?: {} weaksup? {}".format(
                                                                                        args.alpha,
                                                                                        args.beta,
                                                                                        args.seed,
                                                                                        args.context_condition,
                                                                                        args.cuda,
                                                                                        args.weaksup))

    assert args.weaksup.startswith('pretrain')

    # repeat training on same model w/ different random seeds for |num_iter| times
    for i in range(1, args.num_iter + 1):
        print("\nPre-training iteration {}".format(i))
        
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
        train_dataset = ColorDataset(split='Train', context_condition=args.context_condition)
        train_x_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        train_y_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        N_mini_batches = len(train_x_loader)
        vocab_size = train_dataset.vocab_size
        vocab = train_dataset.vocab
        w2i = vocab['w2i']
        pad_index = w2i[PAD_TOKEN]

        # Define test dataset
        test_dataset = ColorDataset(vocab=vocab, split='Validation', context_condition=args.context_condition)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

        # Define latent dimension |z_dim|
        z_dim = args.z_dim

        # Define model
        vae_emb = TextEmbedding(vocab_size)
        vae_rgb_enc = ColorEncoder(z_dim)
        vae_txt_enc = TextEncoder(vae_emb, z_dim)
        vae_rgb_dec = ColorDecoder(z_dim)
        vae_txt_dec = TextDecoder(vae_emb, z_dim, w2i[SOS_TOKEN], w2i[EOS_TOKEN],
                                    w2i[PAD_TOKEN], w2i[UNK_TOKEN], word_dropout=args.dropout)

        # Mount models unto GPU
        vae_emb = vae_emb.to(device)
        vae_rgb_enc = vae_rgb_enc.to(device)
        vae_txt_enc = vae_txt_enc.to(device)
        vae_rgb_dec = vae_rgb_dec.to(device)
        vae_txt_dec = vae_txt_dec.to(device)

        # Define optimizer
        optim_rgb = torch.optim.Adam(
            chain(
            vae_rgb_enc.parameters(),
            vae_rgb_dec.parameters(),
        ), lr=args.lr)

        optim_txt = torch.optim.Adam(
            chain(
            vae_emb.parameters(),
            vae_txt_enc.parameters(),
            vae_txt_dec.parameters(),
        ), lr=args.lr)

        rgb_best_loss = float('inf')
        txt_best_loss = float('inf')
        rgb_track_loss = np.zeros((args.epochs, 2))
        txt_track_loss = np.zeros((args.epochs, 2))
        
        for epoch in range(1, args.epochs + 1):
            rgb_train_loss = train_rgb_only(epoch, optim_rgb)
            txt_train_loss = train_text_only(epoch, optim_txt) if not args.weaksup.endswith('reverse') else train_text_only_rev(epoch, optim_txt)
            txt_test_loss, rgb_test_loss = test(epoch)

            rgb_is_best = rgb_test_loss < rgb_best_loss
            txt_is_best = txt_test_loss < txt_best_loss
            rgb_best_loss = min(rgb_test_loss, rgb_best_loss)
            txt_best_loss = min(txt_test_loss, txt_best_loss)
            rgb_track_loss[epoch - 1, 0] = rgb_train_loss
            rgb_track_loss[epoch - 1, 1] = rgb_test_loss
            txt_track_loss[epoch - 1, 0] = txt_train_loss
            txt_track_loss[epoch - 1, 1] = txt_test_loss
            
            save_checkpoint({
                'epoch': epoch,
                'vae_emb': vae_emb.state_dict(),
                'vae_rgb_enc': vae_rgb_enc.state_dict(),
                'vae_txt_enc': vae_txt_enc.state_dict(),
                'vae_rgb_dec': vae_rgb_dec.state_dict(),
                'vae_txt_dec': vae_txt_dec.state_dict(),
                'optimizer': optim_txt.state_dict(),
                'optimizer': optim_rgb.state_dict(),
                'rgb_track_loss': rgb_track_loss,
                'txt_track_loss': txt_track_loss,
                'cmd_line_args': args,
                'vocab': vocab,
                'vocab_size': vocab_size,
                'seed': random_iter_seed
            }, rgb_is_best or txt_is_best,
            folder=args.out_dir,
            filename='checkpoint_vae_pretrain_{}_alpha={}_beta={}'.format(i, args.alpha, args.beta),
            modality=(rgb_is_best, txt_is_best))
            np.save(os.path.join(args.out_dir,
                'rgb_loss_{}.npy'.format(i)), rgb_track_loss)
            np.save(os.path.join(args.out_dir,
                'txt_loss_{}.npy'.format(i)), txt_track_loss)
