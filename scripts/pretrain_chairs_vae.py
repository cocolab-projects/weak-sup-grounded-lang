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
from utils import (AverageMeter, save_checkpoint, _reparameterize,
                    loss_multimodal, loss_multimodal_only, loss_text_unimodal, loss_image_unimodal)
from models import (TextEmbedding, TextEncoder, TextDecoder,
                    ImageEncoder, ImageTextEncoder, ImageDecoder)
from forward import (forward_vae_image_text, forward_vae_image, forward_vae_text)

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

    print("Called python script: pretrain_chairs_vae.py")

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)

    def train_image_only(epoch, optimizer):
        """Function: train
        Args:
            param1 (int) epoch: training epoch
        Returns:
            (float): training loss over epoch
        """
        vae_img_enc.train()
        vae_img_dec.train()

        models_y = (vae_img_enc, vae_img_dec)
        
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(train_y_loader))
        for batch_idx, (y_img, _, _, _, _, _) in enumerate(train_y_loader):
            batch_size = y_img.size(0)
            y_img = y_img.to(device).float()
            data_y_args = [y_img]

            output_y_dict = forward_vae_image(data_y_args, models_y)
            loss_y = loss_image_unimodal(output_y_dict, batch_size, beta=args.beta, gamma=args.gamma)

            loss_meter.update(loss_y.item(), batch_size)
            optimizer.zero_grad()
            loss_y.backward()
            optimizer.step()

            pbar.set_postfix({'img_loss': loss_meter.avg})
            pbar.update()
        pbar.close()

        if epoch % 10 == 0:
            print('====> Image Train Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
        
        return loss_meter.avg

    def train_text_only(epoch, optimizer):
        vae_emb.train()
        vae_txt_enc.train()
        vae_txt_dec.train()

        models_x = (vae_txt_enc, vae_txt_dec)
        
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(train_x_loader))
        for batch_idx, (_, _, _, x_src, x_tgt, x_len) in enumerate(train_x_loader):
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
        vae_img_enc.eval()
        vae_img_dec.eval()

        with torch.no_grad():
            text_loss_meter = AverageMeter()
            img_loss_meter = AverageMeter()
            pbar = tqdm(total=len(test_loader))

            models_x = (vae_txt_enc, vae_txt_dec)
            models_y = (vae_img_enc, vae_img_dec)

            for batch_idx, (y_img, _, _, x_src, x_tgt, x_len) in enumerate(test_loader):
                batch_size = y_img.size(0)
                y_img = y_img.to(device).float()
                x_tgt = x_tgt.to(device)
                x_src = x_src.to(device)
                x_len = x_len.to(device)

                data_x_args = [x_tgt, x_src, x_len] if args.weaksup.endswith('reverse') else [x_src, x_tgt, x_len]
                data_y_args = [y_img]

                output_x_dict = forward_vae_text(data_x_args, models_x)
                output_x_dict['pad_index'] = pad_index
                output_y_dict = forward_vae_image(data_y_args, models_y)

                loss_x = loss_text_unimodal(output_x_dict, batch_size, alpha=args.alpha, gamma=args.gamma)
                loss_y = loss_image_unimodal(output_y_dict, batch_size, beta=args.beta, gamma=args.gamma)

                text_loss_meter.update(loss_x.item(), batch_size)
                img_loss_meter.update(loss_y.item(), batch_size)

                pbar.update()
            pbar.close()
            if epoch % 10 == 0:
                print('====> Test Epoch: {}\tImage Test Loss: {:.4f} Text Test Loss: {:.4f}'.format(epoch, img_loss_meter.avg, text_loss_meter.avg))
        return text_loss_meter.avg, img_loss_meter.avg

    print("=== begin pretraining ===")
    print(args)

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
        train_dataset = Chairs_ReferenceGame(split='Train', context_condition=args.context_condition)
        train_x_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        train_y_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        N_mini_batches = len(train_x_loader)
        vocab_size = train_dataset.vocab_size
        vocab = train_dataset.vocab
        w2i = vocab['w2i']
        pad_index = w2i[PAD_TOKEN]

        # Define test dataset
        test_dataset = Chairs_ReferenceGame(vocab=vocab, split='Validation', context_condition=args.context_condition)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

        # Define latent dimension |z_dim|
        z_dim = args.z_dim

        # Define model
        channels, img_size = 3, 32
        vae_emb = TextEmbedding(vocab_size)
        vae_img_enc = ImageEncoder(channels, img_size, z_dim)
        vae_txt_enc = TextEncoder(vae_emb, z_dim)
        vae_img_dec = ImageDecoder(channels, img_size, z_dim)
        vae_txt_dec = TextDecoder(vae_emb, z_dim, w2i[SOS_TOKEN], w2i[EOS_TOKEN],
                                    w2i[PAD_TOKEN], w2i[UNK_TOKEN], word_dropout=args.dropout)

        # Mount models unto GPU
        vae_emb = vae_emb.to(device)
        vae_img_enc = vae_img_enc.to(device)
        vae_txt_enc = vae_txt_enc.to(device)
        vae_img_dec = vae_img_dec.to(device)
        vae_txt_dec = vae_txt_dec.to(device)

        # Define optimizer
        optim_img = torch.optim.Adam(
            chain(
            vae_img_enc.parameters(),
            vae_img_dec.parameters(),
        ), lr=args.lr)

        optim_txt = torch.optim.Adam(
            chain(
            vae_emb.parameters(),
            vae_txt_enc.parameters(),
            vae_txt_dec.parameters(),
        ), lr=args.lr)

        img_best_loss = float('inf')
        txt_best_loss = float('inf')
        img_track_loss = np.zeros((args.epochs, 2))
        txt_track_loss = np.zeros((args.epochs, 2))
        
        for epoch in range(1, args.epochs + 1):
            img_train_loss = train_image_only(epoch, optim_img)
            txt_train_loss = train_text_only(epoch, optim_txt) if not args.weaksup.endswith('reverse') else train_text_only_rev(epoch, optim_txt)
            txt_test_loss, img_test_loss = test(epoch)

            img_is_best = img_test_loss < img_best_loss
            txt_is_best = txt_test_loss < txt_best_loss
            img_best_loss = min(img_test_loss, img_best_loss)
            txt_best_loss = min(txt_test_loss, txt_best_loss)
            img_track_loss[epoch - 1, 0] = img_train_loss
            img_track_loss[epoch - 1, 1] = img_test_loss
            txt_track_loss[epoch - 1, 0] = txt_train_loss
            txt_track_loss[epoch - 1, 1] = txt_test_loss
            
            save_checkpoint({
                'epoch': epoch,
                'vae_emb': vae_emb.state_dict(),
                'vae_img_enc': vae_img_enc.state_dict(),
                'vae_txt_enc': vae_txt_enc.state_dict(),
                'vae_img_dec': vae_img_dec.state_dict(),
                'vae_txt_dec': vae_txt_dec.state_dict(),
                'optimizer': optim_txt.state_dict(),
                'optimizer': optim_img.state_dict(),
                'img_track_loss': img_track_loss,
                'txt_track_loss': txt_track_loss,
                'cmd_line_args': args,
                'vocab': vocab,
                'vocab_size': vocab_size,
                'seed': random_iter_seed
            }, img_is_best or txt_is_best,
            folder=args.out_dir,
            filename='checkpoint_vae_pretrain_{}_alpha={}_beta={}'.format(i, args.alpha, args.beta),
            modality=(img_is_best, txt_is_best))
            np.save(os.path.join(args.out_dir,
                'img_loss_{}.npy'.format(i)), img_track_loss)
            np.save(os.path.join(args.out_dir,
                'txt_loss_{}.npy'.format(i)), txt_track_loss)
