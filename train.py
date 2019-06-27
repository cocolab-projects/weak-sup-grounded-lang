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
import color_dataset

from utils import (AverageMeter)
from models import TextEmbedding, Supervised


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='colors (for now)')
    parser.add_argument('out_dir', type=str, help='where to save checkpoints')
    # unnecessary
    parser.add_argument('--d-dim', type=int, default=100,
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
    train_dataset = ColorDataset(train=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    N_mini_batches = len(train_loader)
    vocab_size = len(train_dataset.vocab['w2i'])
    vocab = train_dataset.vocab

    test_dataset = ColorDataset(train=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

    sup_emb = TextEmbedding(vocab_size)
    sup_img = Supervised(sup_emb)
    
    sup_emb = sup_emb.to(device)
    sup_img = sup_img.to(device)

    optimizer = torch.optim.Adam(
        chain(
            sup_emb.parameters(), 
            sup_img.parameters(),
        ), lr=args.lr)


    def train(epoch):
        sup_emb.train()
        sup_img.train()

        # !!! WRITE AverageMeter !!!
        # loss_meter = AverageMeter()

        for batch_idx, (y_rgb, x_inp, x_len) in enumerate(train_loader):
            # ?? what is size(0) doing? size of |x_input|'s 0th dimension ???
            batch_size = x_inp.size(0) 
            y_rgb = y_rgb.to(device)
            x_inp = x_inp.to(device)
            x_len = x_len.to(device)

            # obtain predicted rgb
            pred_rgb = sup_img(x_input, x_len)

            # loss between actual and predicted rgb
            # !!! MIGHT HAVE TO WRIE LOSS FUNCTION separately -- binary cross entropy? !!!
            loss = torch.mean(pred_rgb - y_rgb)

            # loss_meter.update(loss.item(), batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_meter.avg))
            
        print('====> Train Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
        
        return loss_meter.avg


    def test(epoch):
        sup_emb.eval()
        sup_img.eval()

        with torch.no_grad():
            loss_meter = AverageMeter()

            for batch_idx, (y_rgb, x_inp, x_len) in enumerate(test_loader):
                batch_size = x_inp.size(0)
                y_rgb = y_rgb.to(device)
                x_inp = x_inp.to(device)
                x_len = x_len.to(device)

                pred_rgb = sup_img(x)

                loss = torch.mean(pred_rgb - y_rgb)
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
        
        # ??? WRITE SEPARATE |save_checkpoint| function ???
        # save_checkpoint({
        #     'sup_emb': sup_emb.state_dict(),
        #     'sup_img': sup_img.state_dict(),
        #     'cmd_line_args': args,
        #     'vocab': vocab,
        # }, is_best, folder=args.out_dir)
        np.save(os.path.join(args.out_dir, 'loss.npy'), track_loss)