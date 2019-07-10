import os
import sys
import numpy as np
from tqdm import tqdm

import torch 
from torch.utils.data import DataLoader
from utils import (AverageMeter)
from models import (ColorSupervised)
from color_dataset import (ColorDataset, Colors_ReferenceGame, WeakSup_ColorReference)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir', type=str, help='where to load checkpoints from')
    parser.add_argument('out_dir', type=str, help='where to store results from')
    parser.add_argument('sup_lvl', type=float, help='supervision level, if any')
    parser.add_argument('--num_iter', type=int, default=1,
                        help='number of total iterations performed on each setting [default: 1]')
    parser.add_argument('--hard', action='store_true', help='whether the dataset is to be easy')
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # set learning device
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    def test_loss(model, vocab, split='Test'):
        '''
        Test model on newly seen dataset -- gives final test loss
        '''
        assert vocab != None
        print("Computing final test loss on newly seen dataset...")

        test_dataset = ColorDataset(vocab=vocab, split=split, hard=args.hard)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=100)

        model.eval()
        with torch.no_grad():
            loss_meter = AverageMeter()

            for batch_idx, (tgt_rgb, x_inp, x_tgt, x_len) in enumerate(test_loader):
                batch_size = x_inp.size(0)
                tgt_rgb = tgt_rgb.to(device).float()
                x_inp = x_inp.to(device)
                x_len = x_len.to(device)

                # obtain predicted rgb
                pred_rgb = model(x_inp, x_len)

                # loss between actual and predicted rgb: cross entropy
                loss = torch.mean(torch.pow(pred_rgb - tgt_rgb, 2))
                loss_meter.update(loss.item(), batch_size)
            print('====> Final Test Loss: {:.4f}'.format(loss_meter.avg))
        return loss_meter.avg

    def test_refgame_accuracy(model, vocab):
        '''
        Final accuracy: test on reference game dataset
        '''
        assert vocab != None
        print("Computing final accuracy for reference game settings...")

        ref_dataset = Colors_ReferenceGame(vocab, split='Test', hard=args.hard)
        ref_loader = DataLoader(ref_dataset, shuffle=False, batch_size=100)

        model.eval()

        with torch.no_grad():
            total_count = 0
            correct_count = 0
            for batch_idx, (tgt_rgb, d1_rgb, d2_rgb, x_inp, x_tgt, x_len) in enumerate(ref_loader):
                batch_size = x_inp.size(0)
                tgt_rgb = tgt_rgb.to(device).float()
                d1_rgb = d1_rgb.to(device).float()
                d2_rgb = d2_rgb.to(device).float()
                x_inp = x_inp.to(device)
                x_len = x_len.to(device)

                # obtain predicted rgb
                pred_rgb = model(x_inp, x_len)

                # loss between actual and predicted rgb:
                diff_tgt = torch.mean(torch.pow(pred_rgb - tgt_rgb, 2), 1)
                diff_d1 = torch.mean(torch.pow(pred_rgb - d1_rgb, 2), 1)
                diff_d2 = torch.mean(torch.pow(pred_rgb - d2_rgb, 2), 1)

                total_count += diff_tgt.size(0)
                correctList = np.logical_and(np.array(diff_tgt.cpu()) < np.array(diff_d1.cpu()), np.array(diff_tgt.cpu()) < np.array(diff_d2.cpu()))
                correct_count += np.sum(correctList)

            accuracy = correct_count / float(total_count) * 100
            print('====> Final Accuracy: {}/{} = {}%'.format(correct_count, total_count, accuracy))
        return accuracy

    def load_checkpoint(folder='./', filename='model_best'):
        checkpoint = torch.load(folder + filename + '.pth.tar')
        epoch = checkpoint['epoch']
        track_loss = checkpoint['track_loss']
        sup_img = checkpoint['sup_img']
        vocab = checkpoint['vocab']
        vocab_size = checkpoint['vocab_size']
        return epoch, track_loss, sup_img, vocab, vocab_size

    print("=== begin testing ===")

    losses, accuracies = [], []
    for iter_num in range(1, args.num_iter + 1):
        print("loading checkpoint ...")
        epoch, track_loss, sup_img, vocab, vocab_size = \
            load_checkpoint(folder=args.load_dir,
                            filename='checkpoint_{}_{}_best'.format(args.sup_lvl, iter_num))
        print("iteration {}".format(iter_num))
        print("best training epoch: {}".format(epoch))

        txt2img = ColorSupervised(vocab_size)
        txt2img.load_state_dict(sup_img)

        txt2img = txt2img.to(device)

        losses.append(test_loss(txt2img, vocab))
        accuracies.append(test_refgame_accuracy(txt2img, vocab))

    losses = np.array(losses)
    accuracies = np.array(accuracies)
    np.save(os.path.join(args.out_dir, 'accuracies_{}.npy'.format(args.sup_lvl)), accuracies)
    np.save(os.path.join(args.out_dir, 'final_losses_{}.npy'.format(args.sup_lvl)), losses)

    print()
    print("======> Average loss: {}".format(np.mean(losses)))
    print("======> Average accuracy: {}".format(np.mean(accuracies)))
