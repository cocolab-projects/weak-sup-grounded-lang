from models import (TextEmbedding, Supervised)
from utils import (AverageMeter, save_checkpoint, get_text)
from color_dataset import (ColorDataset, Colors_ReferenceGame)
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

def load_checkpoint(folder='./', filename='model_best.pth.tar'):
    checkpoint = torch.load(folder + filename)
    epoch = checkpoint['epoch']
    track_loss = checkpoint['track_loss']
    sup_emb = checkpoint['sup_emb']
    sup_img = checkpoint['sup_img']
    vocab = checkpoint['vocab']
    vocab_size = checkpoint['vocab_size']
    return epoch, track_loss, sup_emb, sup_img, vocab, vocab_size

def test_loss(model, vocab, split='Test'):
    '''
    Test model on newly seen dataset -- gives final test loss
    '''
    assert (vocab != None)
    print("Computing final test loss for newly seen datasets...")

    test_dataset = ColorDataset(vocab=vocab, split=split)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=100)
    N_mini_batches = len(test_loader)

    model.eval()

    with torch.no_grad():
        loss_meter = AverageMeter()

        for batch_idx, (y_rgb, x_inp, x_len) in enumerate(test_loader):
            batch_size = x_inp.size(0)
            y_rgb = y_rgb.float()

            pred_rgb = model(x_inp, x_len)
            pred_rgb = torch.sigmoid(pred_rgb)

            loss = torch.mean(torch.pow(pred_rgb - y_rgb, 2))
            
            given_text = get_text(vocab['i2w'], np.array(x_inp[0]), x_len[0].item())
            pred_RGB = (pred_rgb[0] * 255.0).long().tolist()
            print('{0} matches with text: {1}'.format(pred_RGB, given_text))

            loss_meter.update(loss.item(), batch_size)
        print('====> Final Test Loss: {:.4f}'.format(loss_meter.avg))
        print()

def test_refgame_accuracy(model, vocab):
    '''
    Final accuracy: test on reference game dataset
    '''
    assert (vocab != None)
    print("Computing final accuracy for reference game settings...")

    ref_dataset = Colors_ReferenceGame(vocab, split='Test')
    ref_loader = DataLoader(ref_dataset, shuffle=False, batch_size=100)
    N_mini_batches = len(ref_loader)

    model.eval()

    with torch.no_grad():
        loss_meter = AverageMeter()

        total_count = 0
        correct_count = 0

        for batch_idx, (tgt_rgb, d1_rgb, d2_rgb, x_inp, x_len) in enumerate(ref_loader):
            batch_size = x_inp.size(0)
            tgt_rgb = tgt_rgb.float()
            d1_rgb = d1_rgb.float()
            d2_rgb = d2_rgb.float()

            pred_rgb = model(x_inp, x_len)
            pred_rgb = torch.sigmoid(pred_rgb)

            loss = torch.mean(torch.pow(pred_rgb - tgt_rgb, 2))

            for i in range(batch_size):
                diff_tgt = torch.mean(torch.pow(pred_rgb[i] - tgt_rgb[i], 2))
                diff_d1 = torch.mean(torch.pow(pred_rgb[i] - d1_rgb[i], 2))
                diff_d2 = torch.mean(torch.pow(pred_rgb[i] - d2_rgb[i], 2))
                total_count += 1
                if diff_tgt.item() < diff_d1.item() and diff_tgt.item() < diff_d2.item():
                    correct_count += 1

            loss_meter.update(loss.item(), batch_size)

        accuracy = correct_count / float(total_count) * 100
        print('====> Final Test Loss: {:.4f}'.format(loss_meter.avg))
        print('====> Final Accuracy: {}/{} = {}%'.format(correct_count, total_count, accuracy))
        print()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir', type=str, help='where to load checkpoints from')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

epoch, track_loss, sup_emb, sup_img, vocab, vocab_size = load_checkpoint(folder=args.load_dir,filename='model_best.pth.tar')

emb = TextEmbedding(vocab_size)
emb.load_state_dict(sup_emb)
txt2img = Supervised(emb)
txt2img.load_state_dict(sup_img)

print("best training epoch: {}".format(epoch))
print()
test_loss(txt2img, vocab)
test_refgame_accuracy(txt2img, vocab)
