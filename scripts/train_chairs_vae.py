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
from torchvision import transforms
from torchvision.utils import save_image

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
    parser.add_argument('sup_lvl', type=float, default = 1.0,
                        help='how much of the data to supervise [default: 1.0]')
    parser.add_argument('--dataset', type=str, default='chairs')
    parser.add_argument('--z_dim', type=int, default=100,
                        help='number of latent dimension [default = 100]')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='word dropout in text generation [default = 0.]')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size [default=100]')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='learning rate [default=0.0003]')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs [default: 50]')
    parser.add_argument('--alpha', type=float, default=1,
                        help='lambda argument for text loss')
    parser.add_argument('--beta', type=float, default=1,
                        help='lambda argument for rgb loss')
    parser.add_argument('--gamma', type=float, default=1,
                        help='lambda hyperparameter for D_KL term loss')
    parser.add_argument('--weaksup', type=str, default='default',
                        help='mode for unpaired dataset training')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_iter', type=int, default = 1,
                        help='number of iterations for this setting [default: 1]')
    parser.add_argument('--load_dir', type=str, help='where to load (pretrained) checkpoints from')
    parser.add_argument('--context_condition', type=str, default='far',
                        help='whether the dataset is to include all data')
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    args = parser.parse_args()

    print("Called python script: train_chairs_vae.py")

    if args.dataset == 'chairs':
        from chair_dataset import (Chairs_ReferenceGame, Weaksup_Chairs_Reference)
    if args.dataset == 'critters':
        from critter_dataset import (Critters_ReferenceGame, Weaksup_Critters_Reference)

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
        vae_img_enc.train()
        vae_txt_enc.train()
        vae_mult_enc.train()
        vae_img_dec.train()
        vae_txt_dec.train()

        loss_meter = AverageMeter()
        pbar = tqdm(total=len(train_xy_loader))
        for batch_idx, (tgt_img, _, _, x_src, x_tgt, x_len) in enumerate(train_xy_loader):
            batch_size = x_src.size(0) 
            tgt_img = tgt_img.to(device).float()
            x_src = x_src.to(device)
            x_tgt = x_tgt.to(device)
            x_len = x_len.to(device)

            models_xy = (vae_txt_enc, vae_img_enc, vae_mult_enc, vae_txt_dec, vae_img_dec)
            out = forward_vae_image_text((tgt_img, x_src, x_tgt, x_len), models_xy)
            out['pad_index'] = pad_index

            # compute loss
            if args.weaksup.endswith('2terms'):
                loss = loss_multimodal_only(out, batch_size, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
            else:
                loss = loss_multimodal(out, batch_size, alpha=args.alpha, beta=args.beta, gamma=args.gamma)

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
        """Function: train_weakly_supervised
        Args:
            param1 (int) epoch: training epoch
        Returns:
            (float): training loss over epoch
        """
        vae_emb.train()
        vae_img_enc.train()
        vae_txt_enc.train()
        vae_mult_enc.train()
        vae_img_dec.train()
        vae_txt_dec.train()

        train_xy_iterator = train_xy_loader.__iter__()
        train_x_iterator = train_x_loader.__iter__()
        train_y_iterator = train_y_loader.__iter__()

        models_xy = (vae_txt_enc, vae_img_enc, vae_mult_enc, vae_txt_dec, vae_img_dec)
        models_x = (vae_txt_enc, vae_txt_dec)
        models_y = (vae_img_enc, vae_img_dec)
        
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(train_y_iterator))
        for batch_i in range(len(train_y_iterator)):
            try:
                y_img, _, _, x_src, x_tgt, x_len = next(train_xy_iterator)
                data_xy_args = [y_img, x_src, x_tgt, x_len]
            except StopIteration:
                train_xy_iterator = train_xy_loader.__iter__()
                y_img, _, _, x_src, x_tgt, x_len = next(train_xy_iterator)
                data_xy_args = [y_img, x_src, x_tgt, x_len]

            try:
                _, _, _, x_src, x_tgt, x_len = next(train_x_iterator)
                data_x_args = [x_src, x_tgt, x_len]
            except StopIteration:
                train_x_iterator = train_x_loader.__iter__()
                _, _, _, x_src, x_tgt, x_len = next(train_x_iterator)
                data_x_args = [x_src, x_tgt, x_len]

            try:
                y, _, _, _, _, _ = next(train_y_iterator)
                data_y_args = [y]
            except StopIteration:
                train_y_iterator = train_y_loader.__iter__()
                y, _, _, _, _, _ = next(train_y_iterator)
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

            output_xy_dict = forward_vae_image_text(data_xy_args, models_xy)
            output_x_dict = forward_vae_text(data_x_args, models_x)
            output_y_dict = forward_vae_image(data_y_args, models_y)
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
        vae_img_enc.eval()
        vae_txt_enc.eval()
        vae_mult_enc.eval()
        vae_img_dec.eval()
        vae_txt_dec.eval()

        with torch.no_grad():
            loss_meter = AverageMeter()
            pbar = tqdm(total=len(test_loader))

            for batch_idx, (tgt_img, d1_img, d2_img, x_src, x_tgt, x_len) in enumerate(test_loader):
                batch_size = x_src.size(0) 
                tgt_img = tgt_img.to(device).float()
                x_src = x_src.to(device)
                x_tgt = x_tgt.to(device)
                x_len = x_len.to(device)

                # Encode to |z|
                z_x_mu, z_x_logvar = vae_txt_enc(x_src, x_len)
                z_y_mu, z_y_logvar = vae_img_enc(tgt_img)
                z_xy_mu, z_xy_logvar = vae_mult_enc(tgt_img, x_src, x_len)

                # sample via reparametrization
                z_sample_x = _reparameterize(z_x_mu, z_x_logvar)
                z_sample_y = _reparameterize(z_y_mu, z_y_logvar)
                z_sample_xy = _reparameterize(z_xy_mu, z_xy_logvar)

                # "predictions"
                y_mu_z_y = vae_img_dec(z_sample_y)
                y_mu_z_xy = vae_img_dec(z_sample_xy)
                x_logit_z_x = vae_txt_dec(z_sample_x, x_src, x_len)
                x_logit_z_xy = vae_txt_dec(z_sample_xy, x_src, x_len)

                out = {'z_x_mu': z_x_mu, 'z_x_logvar': z_x_logvar,
                        'z_y_mu': z_y_mu, 'z_y_logvar': z_y_logvar,
                        'z_xy_mu': z_xy_mu, 'z_xy_logvar': z_xy_logvar,
                        'y_mu_z_y': y_mu_z_y, 'y_mu_z_xy': y_mu_z_xy, 
                        'x_logit_z_x': x_logit_z_x, 'x_logit_z_xy': x_logit_z_xy,
                        'y': tgt_img, 'x': x_tgt, 'x_len': x_len, 'pad_index': pad_index}

                # compute loss
                loss = loss_multimodal(out, batch_size, alpha=args.alpha, beta=args.beta)
                loss_meter.update(loss.item(), batch_size)

                pbar.update()
            pbar.close()
            if epoch % 10 == 0:
                print('====> Test Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
        return loss_meter.avg

    def load_pretrained_checkpoint(iter_num, args, folder='./'):
        img_best_filename = 'checkpoint_vae_pretrain_{}_alpha={}_beta={}_rgb_best'.format(iter_num,
                                                                                    args.alpha,
                                                                                    args.beta)
        txt_best_filename = 'checkpoint_vae_pretrain_{}_alpha={}_beta={}_txt_best'.format(iter_num,
                                                                                    args.alpha,
                                                                                    args.beta)
        print("\nloading pretrained checkpoint file:")
        print("{}.pth.tar ...".format(img_best_filename)) 
        print("{}.pth.tar ...\n".format(txt_best_filename))
        print("Post training version {}".format(args.weaksup))

        img_checkpoint = torch.load(os.path.join(folder, img_best_filename + '.pth.tar'))
        txt_checkpoint = torch.load(os.path.join(folder, txt_best_filename + '.pth.tar'))

        img_epoch = img_checkpoint['epoch']
        txt_epoch = txt_checkpoint['epoch']

        vae_img_enc_sd = img_checkpoint['vae_img_enc']
        vae_img_dec_sd = img_checkpoint['vae_img_dec']
        
        vae_emb_sd = txt_checkpoint['vae_emb']
        vae_txt_enc_sd = txt_checkpoint['vae_txt_enc']
        vae_txt_dec_sd = txt_checkpoint['vae_txt_dec']
        
        pre_vocab = img_checkpoint['vocab']
        pre_vocab_size = img_checkpoint['vocab_size']
        args = img_checkpoint['cmd_line_args']

        w2i = pre_vocab['w2i']
        pad_index = w2i[PAD_TOKEN]

        channels, img_size = 3, 32
        vae_emb = TextEmbedding(vocab_size)
        vae_img_enc = ImageEncoder(channels, img_size, z_dim)
        vae_txt_enc = TextEncoder(vae_emb, z_dim)
        vae_img_dec = ImageDecoder(channels, img_size, z_dim)
        vae_txt_dec = TextDecoder(vae_emb, z_dim, w2i[SOS_TOKEN], w2i[EOS_TOKEN],
                                    w2i[PAD_TOKEN], w2i[UNK_TOKEN], word_dropout=args.dropout)

        vae_emb.load_state_dict(vae_emb_sd)
        vae_txt_enc.load_state_dict(vae_txt_enc_sd)
        vae_txt_dec.load_state_dict(vae_txt_dec_sd)
        vae_img_enc.load_state_dict(vae_img_enc_sd)
        vae_img_dec.load_state_dict(vae_img_dec_sd)

        return vae_emb, vae_txt_enc, vae_txt_dec, vae_img_enc, vae_img_dec, pre_vocab


#########################################
# Training script
#########################################

    print("=== begin training ===")
    print(args)

    assert args.weaksup in [
                            'default',
                            '6terms',
                            '4terms',
                            'post-nounp-2terms',
                            'post-nounp-4terms',
                            'coin',
                            'post-4terms',
                            'post-6terms',
                            'post-only-img-4terms',
                            'post-only-img-6terms',
                            'post-only-text-4terms',
                            'post-only-text-6terms',
                            ]

    if args.weaksup.startswith('post'):
        assert args.load_dir != None

    # repeat training on same model w/ different random seeds for |num_iter| times
    for iter_num in range(1, args.num_iter + 1):
        print("\nTraining iteration {} for supervision level {}".format(iter_num, args.sup_lvl))
        
        # set random seeds
        random_iter_seed = random.randint(0, 1000000000)
        print("Random seed set to : {}".format(random_iter_seed))
        torch.cuda.manual_seed(random_iter_seed)
        random.seed(random_iter_seed)
        torch.manual_seed(random_iter_seed)
        np.random.seed(random_iter_seed)

        # set learning device
        args.cuda = args.cuda and torch.cuda.is_available()
        device = torch.device('cuda' if args.cuda else 'cpu')

        # Define training dataset & build vocab
        if args.dataset == 'chairs':
            train_dataset = Weaksup_Chairs_Reference(supervision_level=args.sup_lvl, context_condition=args.context_condition)
        # if args.dataset == 'critters':
        #     image_size = 32
        #     image_transform = transforms.Compose([
        #                                             transforms.Resize(image_size),
        #                                             transforms.CenterCrop(image_size),
        #                                         ])
        #     train_dataset = Weaksup_Critters_Reference(supervision_level=args.sup_lvl, context_condition='all', transform=image_transform)
        train_xy_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=1)
        
        N_mini_batches = len(train_xy_loader)
        vocab_size = train_dataset.vocab_size
        vocab = train_dataset.vocab
        w2i = vocab['w2i']
        pad_index = w2i[PAD_TOKEN]

        # Define test dataset
        if args.dataset == 'chairs':
            test_dataset = Chairs_ReferenceGame(vocab=vocab, split='Validation', context_condition=args.context_condition)
        # if args.dataset == 'critters':
        #     image_size = 32
        #     image_transform = transforms.Compose([
        #                                             transforms.Resize(image_size),
        #                                             transforms.CenterCrop(image_size),
        #                                         ])
        #     test_dataset = Critters_ReferenceGame(vocab=vocab, split='Validation', context_condition=args.context_condition, image_transform=image_transform)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=1)

        if args.weaksup != 'default' and not args.weaksup.startswith('post-nounp'):
            unpaired_dataset = Chairs_ReferenceGame(vocab=vocab, split='Train', context_condition=args.context_condition)
            train_x_loader = DataLoader(unpaired_dataset, shuffle=True, batch_size=args.batch_size, num_workers=1)
            train_y_loader = DataLoader(unpaired_dataset, shuffle=True, batch_size=args.batch_size, num_workers=1)

        print("Dataset preparation complete.\n")

        # Define latent dimension |z_dim|
        z_dim = args.z_dim

        # Define model
        channels, img_size = 3, 32
        vae_emb = TextEmbedding(vocab_size)
        vae_img_enc = ImageEncoder(channels, img_size, z_dim)
        vae_txt_enc = TextEncoder(vae_emb, z_dim)
        vae_mult_enc = ImageTextEncoder(channels, img_size, z_dim, vae_emb)
        vae_img_dec = ImageDecoder(channels, img_size, z_dim)
        vae_txt_dec = TextDecoder(vae_emb, z_dim, w2i[SOS_TOKEN], w2i[EOS_TOKEN],
                                    w2i[PAD_TOKEN], w2i[UNK_TOKEN], word_dropout=args.dropout)

        if args.weaksup.startswith('post'):
            if 'only' not in args.weaksup:
                vae_emb, vae_txt_enc, vae_txt_dec, vae_img_enc, vae_img_dec, pre_vocab = load_pretrained_checkpoint(iter_num, args, folder=args.load_dir)
            elif args.weaksup.startswith('post-only-img'):
                _, _, _, vae_img_enc, vae_img_dec, pre_vocab = load_pretrained_checkpoint(iter_num, args, folder=args.load_dir)
            elif args.weaksup.startswith('post-only-text'):
                vae_emb, vae_txt_enc, vae_txt_dec, _, _, pre_vocab = load_pretrained_checkpoint(iter_num, args, folder=args.load_dir)
            assert pre_vocab == vocab
        vae_mult_enc = ImageTextEncoder(channels, img_size, z_dim, vae_emb)

        # Mount devices unto GPU
        vae_emb = vae_emb.to(device)
        vae_img_enc = vae_img_enc.to(device)
        vae_txt_enc = vae_txt_enc.to(device)
        vae_mult_enc = vae_mult_enc.to(device)
        vae_img_dec = vae_img_dec.to(device)
        vae_txt_dec = vae_txt_dec.to(device)

        # Define optimizer
        optimizer = torch.optim.Adam(
            chain(
            vae_emb.parameters(),
            vae_img_enc.parameters(),
            vae_txt_enc.parameters(),
            vae_mult_enc.parameters(),
            vae_img_dec.parameters(),
            vae_txt_dec.parameters(),
        ), lr=args.lr)

        best_loss = float('inf')
        track_loss = np.zeros((args.epochs, 2))
        
        for epoch in range(1, args.epochs + 1):
            if args.weaksup == 'default' or args.weaksup.startswith('post-nounp'):
                train_loss = train(epoch)
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
                'vae_img_enc': vae_img_enc.state_dict(),
                'vae_txt_enc': vae_txt_enc.state_dict(),
                'vae_mult_enc': vae_mult_enc.state_dict(),
                'vae_img_dec': vae_img_dec.state_dict(),
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

    print(args)


# # Encode to |z|
# z_x_mu, z_x_logvar = vae_txt_enc(x_src, x_len)
# z_y_mu, z_y_logvar = vae_img_enc(tgt_img)
# z_xy_mu, z_xy_logvar = vae_mult_enc(tgt_img, x_src, x_len)

# # sample via reparametrization
# z_sample_x = _reparameterize(z_x_mu, z_x_logvar)
# z_sample_y = _reparameterize(z_y_mu, z_y_logvar)
# z_sample_xy = _reparameterize(z_xy_mu, z_xy_logvar)

# # "predictions"
# y_mu_z_y = vae_img_dec(z_sample_y)
# y_mu_z_xy = vae_img_dec(z_sample_xy)
# x_logit_z_x = vae_txt_dec(z_sample_x, x_src, x_len)
# x_logit_z_xy = vae_txt_dec(z_sample_xy, x_src, x_len)

# out = {'z_x_mu': z_x_mu, 'z_x_logvar': z_x_logvar,
#         'z_y_mu': z_y_mu, 'z_y_logvar': z_y_logvar,
#         'z_xy_mu': z_xy_mu, 'z_xy_logvar': z_xy_logvar,
#         'y_mu_z_y': y_mu_z_y, 'y_mu_z_xy': y_mu_z_xy, 
#         'x_logit_z_x': x_logit_z_x, 'x_logit_z_xy': x_logit_z_xy,
#         'y': tgt_img, 'x': x_tgt, 'x_len': x_len, 'pad_index': pad_index}

