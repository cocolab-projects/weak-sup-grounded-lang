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
from color_dataset import (ColorDataset, WeakSup_ColorDataset)

from utils import (AverageMeter, save_checkpoint)
from models import Supervised

def train(epoch):
    sup_img.train()

    loss_meter = AverageMeter()
    pbar = tqdm(total=len(train_loader))
    for batch_idx, (y_rgb, x_inp, x_len) in enumerate(train_loader):
        batch_size = x_inp.size(0) 
        y_rgb = y_rgb.to(device).float()
        x_inp = x_inp.to(device)
        x_len = x_len.to(device)

        # obtain predicted rgb
        pred_rgb = sup_img(x_inp, x_len)
        pred_rgb = torch.sigmoid(pred_rgb)

        # loss between actual and predicted rgb: Mean Squared Error
        loss = torch.mean(torch.pow(pred_rgb - y_rgb, 2))

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
    sup_img.eval()

    with torch.no_grad():
        loss_meter = AverageMeter()

        pbar = tqdm(total=len(test_loader))
        for batch_idx, (y_rgb, x_inp, x_len) in enumerate(test_loader):
            batch_size = x_inp.size(0)
            y_rgb = y_rgb.to(device).float()
            x_inp = x_inp.to(device)
            x_len = x_len.to(device)

            pred_rgb = sup_img(x_inp, x_len)
            pred_rgb = torch.sigmoid(pred_rgb)

            loss = torch.mean(torch.pow(pred_rgb - y_rgb, 2))

            loss_meter.update(loss.item(), batch_size)
            
            pbar.update()
        pbar.close()
        if epoch % 10 == 0:
            print('====> Test Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
    return loss_meter.avg

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str, help='where to save checkpoints')
    parser.add_argument('--d-dim', type=int, default=100,
                        help='number of hidden dimensions [default: 100]')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size [default=100]')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate [default=0.001]')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs [default: 50]')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='interval to print results [default: 10]')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dataset = ColorDataset(split='Train')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    N_mini_batches = len(train_loader)
    vocab_size = train_dataset.vocab_size
    vocab = train_dataset.vocab

    test_dataset = ColorDataset(vocab=vocab, split='Validation')
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

    sup_img = Supervised(vocab_size)
    sup_img = sup_img.to(device)

    optimizer = torch.optim.Adam(
        chain(
            sup_img.parameters(),
        ), lr=args.lr)

    print("begin training...")
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
            'sup_img': sup_img.state_dict(),
            'optimizer': optimizer.state_dict(),
            'track_loss': track_loss,
            'cmd_line_args': args,
            'vocab': vocab,
            'vocab_size': vocab_size,
        }, is_best, folder=args.out_dir,
                    filename='checkpoint_{}_{}'.format(1.0, 1))
        np.save(os.path.join(args.out_dir, 'loss.npy'), track_loss)
