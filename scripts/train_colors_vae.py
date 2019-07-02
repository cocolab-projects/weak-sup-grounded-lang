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

from utils import (AverageMeter, save_checkpoint)
from models import ColorSupervised

if __name__ == '__main__':
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

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str, help='where to save checkpoints')
    parser.add_argument('sup_lvl', type=float, default = 1.0,
                        help='how much of the data to supervise [default: 1.0]')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size [default=100]')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate [default=0.001]')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs [default: 50]')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_iter', type=int, default = 3,
                        help='number of iterations for this setting [default: 1]')
    parser.add_argument('--hard', action='store_true', help='whether the dataset is to be easy')
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("begin training with supervision level: {} ...".format(args.sup_lvl))
    for i in range(1, args.num_iter + 1):
        print()
        print("Training iteration {} for supervision level {}".format(i, args.sup_lvl))
        
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
        train_dataset = WeakSup_ColorDataset(supervision_level=args.sup_lvl, hard=args.hard)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        N_mini_batches = len(train_loader)
        vocab_size = train_dataset.vocab_size
        vocab = train_dataset.vocab

        # Define test dataset
        test_dataset = ColorDataset(vocab=vocab, split='Validation', hard=args.hard)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

        # Define model
        sup_img = ColorSupervised(vocab_size)
        sup_img = sup_img.to(device)
        optimizer = torch.optim.Adam(sup_img.parameters(), lr=args.lr)

        
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
                'seed': random_iter_seed
            }, is_best, folder=args.out_dir,
            filename='checkpoint_{}_{}'.format(args.sup_lvl, i))
            np.save(os.path.join(args.out_dir,
                'loss_{}_{}.npy'.format(args.sup_lvl, i)), track_loss)


