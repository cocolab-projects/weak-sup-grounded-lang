from models import (TextEmbedding, Supervised)
from utils import (AverageMeter, save_checkpoint, get_text)
from color_dataset import ColorDataset
import os
import sys
import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import shutil
from tqdm import tqdm
from itertools import chain

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir', type=str, help='where to load checkpoints from')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size [default=100]')
    args = parser.parse_args()

def load_checkpoint(folder='./', filename='checkpoint.pth.tar'):
    checkpoint = torch.load(folder + filename)
    epoch = checkpoint['epoch']
    track_loss = checkpoint['track_loss']
    sup_emb = checkpoint['sup_emb']
    sup_img = checkpoint['sup_img']
    vocab = checkpoint['vocab']
    vocab_size = checkpoint['vocab_size']
    return epoch, track_loss, sup_emb, sup_img, vocab, vocab_size

def load_best(folder='./', filename='model_best.pth.tar'):
    checkpoint = torch.load(folder + filename)
    epoch = checkpoint['epoch']
    return epoch

epoch, track_loss, sup_emb, sup_img, vocab, vocab_size = load_checkpoint(folder=args.load_dir,filename='checkpoint.pth.tar')
best_epoch = load_best(folder=args.load_dir,filename='model_best.pth.tar')

emb = TextEmbedding(vocab_size)
emb.load_state_dict(sup_emb)
txt2img = Supervised(emb)
txt2img.load_state_dict(sup_img)

# test on newly seen dataset
test_dataset = ColorDataset(vocab=vocab, split='Test')
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)
N_mini_batches = len(test_loader)

emb.eval()
txt2img.eval()

with torch.no_grad():
    loss_meter = AverageMeter()

    for batch_idx, (y_rgb, x_inp, x_len) in enumerate(test_loader):
        batch_size = x_inp.size(0)
        y_rgb = y_rgb.float()

        pred_rgb = txt2img(x_inp, x_len)
        pred_rgb = torch.sigmoid(pred_rgb)

        loss = torch.mean(torch.pow(pred_rgb - y_rgb, 2))
        
        given_text = get_text(vocab['i2w'], np.array(x_inp[0]), x_len[0].item())
        pred_RGB = (pred_rgb[0] * 255.0).long().tolist()
        print('{0} matches with text: {1}'.format(pred_RGB, given_text))

        loss_meter.update(loss.item(), batch_size)
    print('====> Final Test Loss: {:.4f}'.format(loss_meter.avg))

print(best_epoch)

