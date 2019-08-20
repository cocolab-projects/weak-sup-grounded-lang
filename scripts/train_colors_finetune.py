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
from color_dataset import (Colors_ReferenceGame, WeakSup_ColorReference)

from utils import (AverageMeter, save_checkpoint, _reparameterize, loss_multimodal, loss_text_unimodal, loss_image_unimodal, loss_multimodal_only)
from models import (TextEmbedding, TextEncoder, TextDecoder,
                    ColorEncoder, MultimodalEncoder, ColorDecoder, Finetune_Refgame)
from forward import (forward_vae_rgb_text, forward_vae_rgb, forward_vae_text)
# from loss import (VAE_loss)

SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

if __name__ == '__main__':
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir', type=str, help='where to load checkpoints from')
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
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='lambda hyperparameter for text loss')
    parser.add_argument('--beta', type=float, default=10.0,
                        help='lambda hyperparameter for image loss')
    parser.add_argument('--gamma', type=float, default=1,
                        help='lambda hyperparameter for D_KL term loss')
    parser.add_argument('--weaksup', type=str, default='finetune',
                        help='mode for unpaired dataset training')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_iter', type=int, default = 1,
                        help='number of iterations for this setting [default: 1]')
    parser.add_argument('--context_condition', type=str, default='far', help='whether the dataset is to include all data')
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    args = parser.parse_args()

    print("Called python script: train_colors_finetune.py")

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)

    def train(epoch):
        """Function: train
        train MVAE with supervised datapoints
        Args:
            param1 (int) epoch: training epoch
        Returns:
            (float): training loss over epoch
        """
        vae_emb.train()
        vae_rgb_enc.train()
        vae_txt_enc.train()
        vae_mult_enc.train()
        sup_finetune.train()

        loss_meter = AverageMeter()
        pbar = tqdm(total=len(train_xy_loader))
        for batch_idx, (y_rgb, d1_rgb, d2_rgb, x_src, x_tgt, x_len) in enumerate(train_xy_loader):
            batch_size = y_rgb.size(0)
            y_rgb = y_rgb.to(device).float()
            d1_rgb = d1_rgb.to(device).float()
            d2_rgb = d2_rgb.to(device).float()
            x_src = x_src.to(device)
            x_len = x_len.to(device)

            # obtain embeddings
            z_x, _ = vae_txt_enc(x_src, x_len)
            z_y, _ = vae_rgb_enc(y_rgb)
            z_d1, _ = vae_rgb_enc(d1_rgb)
            z_d2, _ = vae_rgb_enc(d2_rgb)
            z_xy, _ = vae_mult_enc(y_rgb, x_src, x_len)

            # obtain predicted compatibility score
            tgt_score = sup_finetune(z_x, z_y)
            d1_score = sup_finetune(z_x, z_d1)
            d2_score = sup_finetune(z_x, z_d2)

            # loss: cross entropy
            loss = F.cross_entropy(torch.cat([tgt_score, d1_score, d2_score], 1), torch.LongTensor(np.zeros(batch_size)).to(device))

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
        sup_finetune.eval()

        with torch.no_grad():
            loss_meter = AverageMeter()
            pbar = tqdm(total=len(test_loader))

            for batch_idx, (y_rgb, d1_rgb, d2_rgb, x_src, x_tgt, x_len) in enumerate(test_loader):
                batch_size = y_rgb.size(0)
                y_rgb = y_rgb.to(device).float()
                d1_rgb = d1_rgb.to(device).float()
                d2_rgb = d2_rgb.to(device).float()
                x_src = x_src.to(device)
                x_len = x_len.to(device)

                # obtain embeddings
                z_x, _ = vae_txt_enc(x_src, x_len)
                z_y, _ = vae_rgb_enc(y_rgb)
                z_d1, _ = vae_rgb_enc(d1_rgb)
                z_d2, _ = vae_rgb_enc(d2_rgb)
                z_xy, _ = vae_mult_enc(y_rgb, x_src, x_len)

                # obtain predicted compatibility score
                tgt_score = sup_finetune(z_x, z_y)
                d1_score = sup_finetune(z_x, z_d1)
                d2_score = sup_finetune(z_x, z_d2)

                # loss: cross entropy
                loss = F.cross_entropy(torch.cat([tgt_score, d1_score, d2_score], 1), torch.LongTensor(np.zeros(batch_size)).to(device))
                loss_meter.update(loss.item(), batch_size)

                pbar.update()
            pbar.close()
            if epoch % 10 == 0:
                print('====> Test Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
        return loss_meter.avg

    def load_vae_checkpoint(iter_num, args, folder='./'):
        best_filename = 'checkpoint_vae_{}_{}_alpha={}_beta={}_best'.format(args.sup_lvl,
                                                                        iter_num,
                                                                        args.alpha,
                                                                        args.beta)
        print("\nloading trained checkpoint file:")
        print("{}.pth.tar ...".format(best_filename))
        print("Post training version {}".format(args.weaksup))

        best_checkpoint = torch.load(os.path.join(folder, best_filename + '.pth.tar'))

        best_epoch = best_checkpoint['epoch']

        vae_rgb_enc_sd = best_checkpoint['vae_rgb_enc']
        vae_emb_sd = best_checkpoint['vae_emb']
        vae_txt_enc_sd = best_checkpoint['vae_txt_enc']
        vae_mult_enc_sd = best_checkpoint['vae_mult_enc']

        pre_vocab = best_checkpoint['vocab']
        pre_vocab_size = best_checkpoint['vocab_size']
        args = best_checkpoint['cmd_line_args']

        w2i = pre_vocab['w2i']
        pad_index = w2i[PAD_TOKEN]

        vae_emb = TextEmbedding(pre_vocab_size)
        vae_txt_enc = TextEncoder(vae_emb, args.z_dim)
        vae_rgb_enc = ColorEncoder(args.z_dim)
        vae_mult_enc = MultimodalEncoder(vae_emb, args.z_dim)

        vae_emb.load_state_dict(vae_emb_sd)
        vae_txt_enc.load_state_dict(vae_txt_enc_sd)
        vae_rgb_enc.load_state_dict(vae_rgb_enc_sd)
        vae_mult_enc.load_state_dict(vae_mult_enc_sd)

        return vae_emb, vae_txt_enc, vae_rgb_enc, vae_mult_enc, pre_vocab


#########################################
# Training script
#########################################

    print("=== begin training ===")
    print(args)

    assert args.weaksup in [
                            'finetune',
                            ]

    assert args.load_dir != None

    # repeat training on same model w/ different random seeds for |num_iter| times
    for iter_num in range(1, args.num_iter + 1):
        print("\nTraining iteration {} for supervision level {}".format(iter_num, args.sup_lvl))
        
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
        print("Initialize datasets for supervised datapoints...")
        train_dataset = WeakSup_ColorReference(supervision_level=args.sup_lvl, context_condition=args.context_condition)
        train_xy_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        N_mini_batches = len(train_xy_loader)
        vocab_size = train_dataset.vocab_size
        vocab = train_dataset.vocab
        w2i = vocab['w2i']
        pad_index = w2i[PAD_TOKEN]

        # Define test dataset
        test_dataset = Colors_ReferenceGame(vocab=vocab, split='Validation', context_condition=args.context_condition)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

        # Define latent dimension |z_dim|
        z_dim = args.z_dim

        # Define model
        sup_finetune = Finetune_Refgame(z_dim=args.z_dim)
        vae_emb = TextEmbedding(vocab_size)
        vae_rgb_enc = ColorEncoder(z_dim)
        vae_txt_enc = TextEncoder(vae_emb, z_dim)

        vae_emb, vae_txt_enc, vae_rgb_enc, vae_mult_enc, pre_vocab = load_vae_checkpoint(iter_num, args, folder=args.load_dir)
        assert pre_vocab == vocab

        # Mount models unto GPU
        sup_finetune = sup_finetune.to(device)
        vae_emb = vae_emb.to(device)
        vae_rgb_enc = vae_rgb_enc.to(device)
        vae_txt_enc = vae_txt_enc.to(device)
        vae_mult_enc = vae_mult_enc.to(device)

        # Define optimizer
        optimizer = torch.optim.Adam(
            chain(
            sup_finetune.parameters(),
        ), lr=args.lr)

        best_loss = float('inf')
        track_loss = np.zeros((args.epochs, 2))
        
        for epoch in range(1, args.epochs + 1):
            train_loss = train(epoch)
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
                'sup_finetune': sup_finetune.state_dict(),
                'optimizer': optimizer.state_dict(),
                'track_loss': track_loss,
                'cmd_line_args': args,
                'vocab': vocab,
                'vocab_size': vocab_size,
                'seed': random_iter_seed
            }, is_best, folder=args.out_dir,
            filename='checkpoint_vae_{}_{}_alpha={}_beta={}'.format(args.sup_lvl, iter_num, args.alpha, args.beta))
            np.save(os.path.join(args.out_dir,
                'loss_{}_{}.npy'.format(args.sup_lvl, iter_num)), track_loss)

    
    print(args)
    print("=== training complete ===")
