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
# from loss import (VAE_loss)

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
    parser.add_argument('--beta', type=float, default=10,
                        help='lambda hyperparameter for image loss')
    parser.add_argument('--gamma', type=float, default=1,
                        help='lambda hyperparameter for D_KL term loss')
    parser.add_argument('--weaksup', type=str, default='default',
                        help='mode for unpaired dataset training')
    parser.add_argument('--load_dir', type=str, help='where to load (pretrained) checkpoints from')
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
        vae_rgb_dec.train()
        vae_txt_dec.train()

        loss_meter = AverageMeter()
        pbar = tqdm(total=len(train_xy_loader))
        for batch_idx, (y_rgb, x_src, x_tgt, x_len) in enumerate(train_xy_loader):
            batch_size = y_rgb.size(0)
            y_rgb = y_rgb.to(device).float()
            x_src = x_src.to(device)
            x_tgt = x_tgt.to(device)
            x_len = x_len.to(device)

            models_xy = (vae_txt_enc, vae_rgb_enc, vae_mult_enc, vae_txt_dec, vae_rgb_dec)
            out = forward_vae_rgb_text((y_rgb, x_src, x_tgt, x_len), models_xy)
            out['pad_index'] = pad_index

            # compute loss
            if args.weaksup.endswith('2terms'):
                loss = loss_multimodal_only(out, batch_size, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
            else:
                loss = loss_multimodal(out, batch_size, alpha=args.alpha, beta=args.beta, gamma=args.gamma)

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

    def train_coin(epoch):
        """Function: train_weakly_supervised
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

        models_xy = (vae_txt_enc, vae_rgb_enc, vae_mult_enc, vae_txt_dec, vae_rgb_dec)
        models_x = (vae_txt_enc, vae_txt_dec)
        models_y = (vae_rgb_enc, vae_rgb_dec)
        
        supervised_loss_meter = AverageMeter()
        unpaired_loss_meter = AverageMeter()
        supervision_info = []
        pbar = tqdm(total=len(train_xy_iterator))
        for batch_idx, (y_rgb, x_src, x_tgt, x_len) in enumerate(train_xy_loader):
            batch_size = y_rgb.size(0)
            y_rgb = y_rgb.to(device).float()
            x_src = x_src.to(device)
            x_tgt = x_tgt.to(device)
            x_len = x_len.to(device)

            supervised = np.random.binomial(1, args.sup_lvl)
            supervision_info.append(supervised)
            if supervised:
                out = forward_vae_rgb_text((y_rgb, x_src, x_tgt, x_len), models_xy)
                out['pad_index'] = pad_index

                loss = loss_multimodal(out, batch_size, alpha=args.alpha, beta=args.beta, gamma=args.gamma)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                supervised_loss_meter.update(loss.item(), batch_size)
            else:
                data_x_args = [x_src, x_tgt, x_len]
                data_y_args = [y_rgb]

                output_y_dict = forward_vae_rgb(data_y_args, models_y)
                output_x_dict = forward_vae_text(data_x_args, models_x)
                output_x_dict['pad_index'] = pad_index

                loss_y = loss_image_unimodal(output_y_dict, batch_size, beta=args.beta, gamma=args.gamma)
                optim_rgb.zero_grad()
                loss_y.backward()
                optim_rgb.step()

                loss_x = loss_text_unimodal(output_x_dict, batch_size, alpha=args.alpha, gamma=args.gamma)
                optim_txt.zero_grad()
                loss_x.backward()
                optim_txt.step()

                loss = loss_x + loss_y
                unpaired_loss_meter.update(loss.item(), batch_size)

            pbar.set_postfix({'sup_loss': supervised_loss_meter.avg, 'unp_loss': unpaired_loss_meter.avg,
                                'supervision_level': sum(supervision_info) / len(supervision_info)})
            pbar.update()
        pbar.close()
        
        if epoch % 10 == 0:
            print('====> Train Epoch: {}\tLoss: {:.4f}'.format(epoch, supervised_loss_meter.avg))
        
        return supervised_loss_meter.avg

    def train_weakly_supervised(epoch):
        """Function: train_weakly_supervised
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
                y_rgb, x_src, x_tgt, x_len = next(train_xy_iterator)
                data_xy_args = [y_rgb, x_src, x_tgt, x_len]
            except StopIteration:
                train_xy_iterator = train_xy_loader.__iter__()
                y_rgb, x_src, x_tgt, x_len = next(train_xy_iterator)
                data_xy_args = [y_rgb, x_src, x_tgt, x_len]

            try:
                _, x_src, x_tgt, x_len = next(train_x_iterator)
                data_x_args = [x_src, x_tgt, x_len]
            except StopIteration:
                train_x_iterator = train_x_loader.__iter__()
                _, x_src, x_tgt, x_len = next(train_x_iterator)
                data_x_args = [x_src, x_tgt, x_len]

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

            if args.weaksup.endswith('4terms'):
                loss_xy = loss_multimodal_only(output_xy_dict, batch_size, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
            if args.weaksup.endswith('6terms'):
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

            # models_xy = (vae_txt_enc, vae_rgb_enc, vae_mult_enc, vae_txt_dec, vae_rgb_dec)
            # models_x = (vae_txt_enc, vae_txt_dec)
            # models_y = (vae_rgb_enc, vae_rgb_dec)

            for batch_idx, (y_rgb, x_src, x_tgt, x_len) in enumerate(test_loader):
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

    def load_pretrained_checkpoint(iter_num, args, folder='./'):
        rgb_best_filename = 'checkpoint_vae_pretrain_{}_alpha={}_beta={}_rgb_best'.format(iter_num,
                                                                                    args.alpha,
                                                                                    args.beta)
        txt_best_filename = 'checkpoint_vae_pretrain_{}_alpha={}_beta={}_txt_best'.format(iter_num,
                                                                                    args.alpha,
                                                                                    args.beta)
        print("\nloading pretrained checkpoint file:")
        print("{}.pth.tar ...".format(rgb_best_filename)) 
        print("{}.pth.tar ...\n".format(txt_best_filename))
        print("Post training version {}".format(args.weaksup))

        rgb_checkpoint = torch.load(os.path.join(folder, rgb_best_filename + '.pth.tar'))
        txt_checkpoint = torch.load(os.path.join(folder, txt_best_filename + '.pth.tar'))

        rgb_epoch = rgb_checkpoint['epoch']
        txt_epoch = txt_checkpoint['epoch']

        vae_rgb_enc_sd = rgb_checkpoint['vae_rgb_enc']
        vae_rgb_dec_sd = rgb_checkpoint['vae_rgb_dec']
        
        vae_emb_sd = txt_checkpoint['vae_emb']
        vae_txt_enc_sd = txt_checkpoint['vae_txt_enc']
        vae_txt_dec_sd = txt_checkpoint['vae_txt_dec']
        
        pre_vocab = rgb_checkpoint['vocab']
        pre_vocab_size = rgb_checkpoint['vocab_size']
        args = rgb_checkpoint['cmd_line_args']

        w2i = pre_vocab['w2i']
        pad_index = w2i[PAD_TOKEN]

        vae_emb = TextEmbedding(pre_vocab_size)
        vae_txt_enc = TextEncoder(vae_emb, args.z_dim)
        vae_txt_dec = TextDecoder(vae_emb, args.z_dim, w2i[SOS_TOKEN], w2i[EOS_TOKEN],
                                    w2i[PAD_TOKEN], w2i[UNK_TOKEN], word_dropout=args.dropout)
        vae_rgb_dec = ColorDecoder(args.z_dim)
        vae_rgb_enc = ColorEncoder(args.z_dim)

        vae_emb.load_state_dict(vae_emb_sd)
        vae_txt_enc.load_state_dict(vae_txt_enc_sd)
        vae_txt_dec.load_state_dict(vae_txt_dec_sd)
        vae_rgb_enc.load_state_dict(vae_rgb_enc_sd)
        vae_rgb_dec.load_state_dict(vae_rgb_dec_sd)

        return vae_emb, vae_txt_enc, vae_txt_dec, vae_rgb_enc, vae_rgb_dec, pre_vocab


#########################################
# Training script
#########################################

    print("=== begin training ===")
    print("args: sup_lvl: {} alpha: {} beta: {} seed: {} context condition?: {} cuda?: {} weaksup: {}".format(args.sup_lvl,
                                                                                    args.alpha,
                                                                                    args.beta,
                                                                                    args.seed,
                                                                                    args.context_condition,
                                                                                    args.cuda,
                                                                                    args.weaksup))

    assert args.weaksup in [
                            'default',
                            '6terms',
                            '4terms',
                            'post-nounp-2terms',
                            'post-nounp-4terms',
                            'coin',
                            'post-4terms',
                            'post-6terms',
                            'post-only-rgb-4terms',
                            'post-only-rgb-6terms',
                            'post-only-text-4terms',
                            'post-only-text-6terms',
                            ]
    
    if args.weaksup.startswith('post'):
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
        if args.weaksup != 'coin':
            train_dataset = WeakSup_ColorDataset(supervision_level=args.sup_lvl, context_condition=args.context_condition)
        else:
            train_dataset = ColorDataset(split='Train', context_condition=args.context_condition)
        train_xy_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        N_mini_batches = len(train_xy_loader)
        vocab_size = train_dataset.vocab_size
        vocab = train_dataset.vocab
        w2i = vocab['w2i']
        pad_index = w2i[PAD_TOKEN]

        if args.weaksup not in ('default', 'coin') and not args.weaksup.startswith('post-nounp'):
            print("Initialize datasets for unpaired datapoints...")
            unpaired_dataset = ColorDataset(vocab=vocab, split='Train', context_condition=args.context_condition)
            train_x_loader = DataLoader(unpaired_dataset, shuffle=True, batch_size=args.batch_size)
            train_y_loader = DataLoader(unpaired_dataset, shuffle=True, batch_size=args.batch_size)

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

        if args.weaksup.startswith('post'):
            if 'only' not in args.weaksup:
                vae_emb, vae_txt_enc, vae_txt_dec, vae_rgb_enc, vae_rgb_dec, pre_vocab = load_pretrained_checkpoint(iter_num, args, folder=args.load_dir)
            elif args.weaksup.startswith('post-only-rgb'):
                _, _, _, vae_rgb_enc, vae_rgb_dec, pre_vocab = load_pretrained_checkpoint(iter_num, args, folder=args.load_dir)
            elif args.weaksup.startswith('post-only-text'):
                vae_emb, vae_txt_enc, vae_txt_dec, _, _, pre_vocab = load_pretrained_checkpoint(iter_num, args, folder=args.load_dir)
            assert pre_vocab == vocab
        vae_mult_enc = MultimodalEncoder(vae_emb, z_dim)

        # Mount models unto GPU
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

        best_loss = float('inf')
        track_loss = np.zeros((args.epochs, 2))
        
        for epoch in range(1, args.epochs + 1):
            if args.weaksup == 'default' or args.weaksup.startswith('post-nounp'):
                train_loss = train(epoch)
            elif args.weaksup == 'coin':
                train_loss = train_coin(epoch)
            else:
                train_loss = train_weakly_supervised(epoch)
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
            filename='checkpoint_vae_{}_{}_alpha={}_beta={}'.format(args.sup_lvl, iter_num, args.alpha, args.beta))
            np.save(os.path.join(args.out_dir,
                'loss_{}_{}.npy'.format(args.sup_lvl, iter_num)), track_loss)

    print("\nargs:: sup_lvl: {} alpha: {} beta: {} seed: {} context condition?: {} cuda?: {} weaksup: {}".format(args.sup_lvl,
                                                                                    args.alpha,
                                                                                    args.beta,
                                                                                    args.seed,
                                                                                    args.context_condition,
                                                                                    args.cuda,
                                                                                    args.weaksup))
    print("=== training complete ===")
