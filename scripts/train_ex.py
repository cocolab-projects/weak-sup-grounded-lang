from __future__ import print_function

import os
import sys
import numpy as np
from tqdm import tqdm
from itertools import chain

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from imagetext_datasets import load_datasets, get_data_dir

from utils import (
    AverageMeter, save_checkpoint,
    bernoulli_log_pdf, text_reconstruction,
    reparameterize, _kl_normal_normal)
from dc_models import VAE_ImageEncoder
from text_models import TextEmbedding, TextDecoder


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='mnist_math|mnist_easy_math|shapeworld|shapeworld_two|coco')
    parser.add_argument('out_dir', type=str, help='where to save checkpoints')
    parser.add_argument('--z-dim', type=int, default=100,
                        help='number of hidden dimensions [default: 100]')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size [default=100]')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate [default=0.0002]')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs [default: 200]')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='interval to print results [default: 10]')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if args.cuda else 'cpu')

    data_dir = get_data_dir(args.dataset)
    train_dataset, n_channels = load_datasets(
        data_dir, args.dataset, vocab=None, gan_standardize=False, train=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    N_mini_batches = len(train_loader)
    vocab_size = train_dataset.vocab_size
    vocab = train_dataset.vocab

    test_dataset, _ = load_datasets(
        data_dir, args.dataset, vocab=vocab, gan_standardize=False, train=False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

    vae_inf = VAE_ImageEncoder(n_channels, 32, args.z_dim)
    vae_emb = TextEmbedding(vocab_size)
    vae_gen = TextDecoder(vae_emb, args.z_dim, train_dataset.sos_index,
                          train_dataset.eos_index, train_dataset.pad_index, 
                          train_dataset.unk_index, hidden_dim=512, 
                          word_dropout=0.75, embedding_dropout=0.5)
    
    vae_emb = vae_emb.to(device)
    vae_inf = vae_inf.to(device)
    vae_gen = vae_gen.to(device)

    optimizer = torch.optim.Adam(
        chain(
            vae_gen.parameters(), 
            vae_inf.parameters(),
            vae_emb.parameters(),
        ), lr=args.lr)


    def train(epoch):
        vae_inf.train()
        vae_gen.train()
        vae_emb.train()

        loss_meter = AverageMeter()

        for batch_idx, (x, y_src, y_tgt, y_len, _) in enumerate(train_loader):
            batch_size = x.size(0)
            x = x.view(batch_size, n_channels, 32, 32)
            x = x.to(device)
            y_src = y_src.to(device)
            y_tgt = y_tgt.to(device)
            y_len = y_len.to(device)

            # define q(z|x)
            z_mu, _ = vae_inf(x)

            # define p(y|z_xy)
            y_tgt_logits = vae_gen(z_mu, y_src, y_len)

            # deterministic loss
            log_p_y_given_z = text_reconstruction(y_tgt, y_tgt_logits, train_dataset.pad_index)
            loss = torch.mean(log_p_y_given_z)

            loss_meter.update(loss.item(), batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_meter.avg))
            
        print('====> Train Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
        
        return loss_meter.avg


    def test(epoch):
        vae_inf.eval()
        vae_gen.eval()
        vae_emb.eval()

        with torch.no_grad():
            loss_meter = AverageMeter()

            for batch_idx, (x, y_src, y_tgt, y_len, _) in enumerate(test_loader):
                batch_size = x.size(0)
                x = x.view(batch_size, n_channels, 32, 32)
                x = x.to(device)
                y_src = y_src.to(device)
                y_tgt = y_tgt.to(device)
                y_len = y_len.to(device)

                z_mu, _ = vae_inf(x)
                y_tgt_logits = vae_gen(z_mu, y_src, y_len)

                log_p_y_given_z = text_reconstruction(y_tgt, y_tgt_logits, train_dataset.pad_index)
                loss = torch.mean(log_p_y_given_z)
                loss_meter.update(loss.item(), batch_size)

            print('====> Test Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
                    
        return loss_meter.avg


    best_loss = sys.maxint
    track_loss = np.zeros((args.epochs, 2))
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)
        track_loss[epoch - 1, 0] = train_loss
        track_loss[epoch - 1, 1] = test_loss
        
        save_checkpoint({
            'vae_inf': vae_inf.state_dict(),
            'vae_gen': vae_gen.state_dict(),
            'vae_emb': vae_emb.state_dict(),
            'cmd_line_args': args,
            'vocab': vocab,
        }, is_best, folder=args.out_dir)
        np.save(os.path.join(args.out_dir, 'loss.npy'), track_loss)