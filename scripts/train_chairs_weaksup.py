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
from chair_dataset import (Chairs_ReferenceGame, Weaksup_Chairs_Reference)

from utils import (AverageMeter, save_checkpoint)
from models import TextImageCompatibility

if __name__ == '__main__':
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
    parser.add_argument('--context_condition', type=str, default='all',
                        help='whether the dataset is to include all data')
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)

    def train(epoch):
        txt_img_comp.train()

        loss_meter = AverageMeter()
        pbar = tqdm(total=len(train_loader))
        for batch_idx, (tgt_chair, d1_chair, d2_chair, x_tgt, x_inp, x_len) in enumerate(train_loader):
            batch_size = x_inp.size(0) 
            tgt_chair = tgt_chair.to(device).float()
            d1_chair = d1_chair.to(device).float()
            d2_chair = d2_chair.to(device).float()
            x_inp = x_inp.to(device)
            x_len = x_len.to(device)

            # obtain predicted rgb
            tgt_score = txt_img_comp(tgt_chair, x_inp, x_len)
            d1_score = txt_img_comp(d1_chair, x_inp, x_len)
            d2_score = txt_img_comp(d2_chair, x_inp, x_len)

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
        txt_img_comp.eval()

        with torch.no_grad():
            loss_meter = AverageMeter()

            pbar = tqdm(total=len(test_loader))
            for batch_idx, (tgt_chair, d1_chair, d2_chair, x_tgt, x_inp, x_len) in enumerate(test_loader):
                batch_size = x_inp.size(0) 
                tgt_chair = tgt_chair.to(device).float()
                d1_chair = d1_chair.to(device).float()
                d2_chair = d2_chair.to(device).float()
                x_inp = x_inp.to(device)
                x_len = x_len.to(device)

                # obtain predicted rgb
                tgt_score = txt_img_comp(tgt_chair, x_inp, x_len)
                d1_score = txt_img_comp(d1_chair, x_inp, x_len)
                d2_score = txt_img_comp(d2_chair, x_inp, x_len)

                # loss between actual and predicted rgb: cross entropy
                loss = F.cross_entropy(torch.cat([tgt_score,d1_score,d2_score], 1), torch.LongTensor(np.zeros(batch_size)).to(device))
                loss_meter.update(loss.item(), batch_size)

                pbar.update()
            pbar.close()
            if epoch % 10 == 0:
                print('====> Test Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
        return loss_meter.avg

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
        train_dataset = Weaksup_Chairs_Reference(supervision_level=args.sup_lvl, split='Train', context_condition=args.context_condition)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        N_mini_batches = len(train_loader)
        vocab_size = train_dataset.vocab_size
        vocab = train_dataset.vocab

        # Define test dataset
        test_dataset = Chairs_ReferenceGame(vocab=vocab, split='Validation', context_condition=args.context_condition)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

        # Define model
        txt_img_comp = TextImageCompatibility(vocab_size)
        optimizer = torch.optim.Adam(txt_img_comp.parameters(), lr=args.lr)
        txt_img_comp.to(device)
        
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
                'txt_img_comp': txt_img_comp.state_dict(),
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


