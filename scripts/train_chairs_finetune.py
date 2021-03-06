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
                    ImageEncoder, ImageTextEncoder, ImageDecoder, Finetune_Refgame)
from forward import (forward_vae_image_text, forward_vae_image, forward_vae_text)

SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

if __name__ == '__main__':
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir', type=str, help='where to load vae embedding checkpoints from')
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
                        help='number of training epochs [default: 100]')
    parser.add_argument('--alpha', type=float, default=1,
                        help='lambda argument for text loss')
    parser.add_argument('--beta', type=float, default=1,
                        help='lambda argument for image loss')
    parser.add_argument('--gamma', type=float, default=1,
                        help='lambda hyperparameter for D_KL term loss')
    parser.add_argument('--weaksup', type=str, default='default',
                        help='mode for unpaired dataset training')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_iter', type=int, default = 1,
                        help='number of iterations for this setting [default: 1]')
    parser.add_argument('--context_condition', type=str, default='far',
                        help='whether the dataset is to include all data')
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    args = parser.parse_args()

    print("Called python script: train_chairs_finetune.py")

    if args.dataset == 'chairs':
        from chair_dataset import (Chairs_ReferenceGame, Weaksup_Chairs_Reference)
    # if args.dataset == 'critters':
    #     from critter_dataset import (Critters_ReferenceGame, Weaksup_Critters_Reference)

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
        sup_finetune.train()

        loss_meter = AverageMeter()
        pbar = tqdm(total=len(train_xy_loader))
        for batch_idx, (tgt_img, d1_img, d2_img, x_src, x_tgt, x_len) in enumerate(train_xy_loader):
            batch_size = tgt_img.size(0)
            tgt_img = tgt_img.to(device).float()
            d1_img = d1_img.to(device).float()
            d2_img = d2_img.to(device).float()
            x_src = x_src.to(device)
            x_len = x_len.to(device)

            # obtain embeddings
            z_x, _ = vae_txt_enc(x_src, x_len)
            z_y, _ = vae_img_enc(tgt_img)
            z_d1, _ = vae_img_enc(d1_img)
            z_d2, _ = vae_img_enc(d2_img)
            # z_xy, _ = vae_mult_enc(y_img, x_src, x_len)

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
        vae_img_enc.eval()
        vae_txt_enc.eval()
        vae_mult_enc.eval()
        sup_finetune.eval()

        with torch.no_grad():
            loss_meter = AverageMeter()
            pbar = tqdm(total=len(test_loader))

            for batch_idx, (tgt_img, d1_img, d2_img, x_src, x_tgt, x_len) in enumerate(test_loader):
                batch_size = tgt_img.size(0)
                tgt_img = tgt_img.to(device).float()
                d1_img = d1_img.to(device).float()
                d2_img = d2_img.to(device).float()
                x_src = x_src.to(device)
                x_len = x_len.to(device)

                # obtain embeddings
                z_x, _ = vae_txt_enc(x_src, x_len)
                z_y, _ = vae_img_enc(tgt_img)
                z_d1, _ = vae_img_enc(d1_img)
                z_d2, _ = vae_img_enc(d2_img)
                # z_xy, _ = vae_mult_enc(y_img, x_src, x_len)

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

        vae_img_enc_sd = best_checkpoint['vae_img_enc']
        vae_emb_sd = best_checkpoint['vae_emb']
        vae_txt_enc_sd = best_checkpoint['vae_txt_enc']
        vae_mult_enc_sd = best_checkpoint['vae_mult_enc']

        pre_vocab = best_checkpoint['vocab']
        pre_vocab_size = best_checkpoint['vocab_size']
        args = best_checkpoint['cmd_line_args']

        w2i = pre_vocab['w2i']
        pad_index = w2i[PAD_TOKEN]

        channels, img_size = 3, 32
        vae_emb = TextEmbedding(vocab_size)
        vae_img_enc = ImageEncoder(channels, img_size, z_dim)
        vae_txt_enc = TextEncoder(vae_emb, z_dim)
        vae_mult_enc = ImageTextEncoder(channels, img_size, z_dim, vae_emb)
        
        vae_emb.load_state_dict(vae_emb_sd)
        vae_txt_enc.load_state_dict(vae_txt_enc_sd)
        vae_img_enc.load_state_dict(vae_img_enc_sd)
        vae_mult_enc.load_state_dict(vae_mult_enc_sd)

        return vae_emb, vae_txt_enc, vae_img_enc, vae_mult_enc, pre_vocab


#########################################
# Training script
#########################################

    print("=== begin training ===")
    print(args)

    assert args.weaksup in ['finetune']
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

        print("Dataset preparation complete.\n")

        # Define latent dimension |z_dim|
        z_dim = args.z_dim

        # Define model
        sup_finetune = Finetune_Refgame(z_dim=args.z_dim)

        vae_emb, vae_txt_enc, vae_img_enc, vae_mult_enc, pre_vocab = load_vae_checkpoint(iter_num, args, folder=args.load_dir)
        assert pre_vocab == vocab

        for param in vae_emb.parameters():
            param.requires_grad = False
        for param in vae_txt_enc.parameters():
            param.requires_grad = False
        for param in vae_img_enc.parameters():
            param.requires_grad = False
        for param in vae_mult_enc.parameters():
            param.requires_grad = False

        # Mount models unto GPU
        sup_finetune = sup_finetune.to(device)
        vae_emb = vae_emb.to(device)
        vae_img_enc = vae_img_enc.to(device)
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
                'vae_img_enc': vae_img_enc.state_dict(),
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

