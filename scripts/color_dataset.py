from __future__ import print_function

import os
import json
import torch 
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import colorsys

from utils import (hsl2rgb, preprocess_text)

import nltk
from nltk import sent_tokenize, word_tokenize

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer

from collections import defaultdict

FILE_DIR = os.path.realpath(os.path.dirname(__file__))
RAW_DIR = os.path.realpath(os.path.join(FILE_DIR, '/mnt/fs5/hokysung/datasets/'))

SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
TRAINING_PERCENTAGE = 64 / 100
TESTING_PERCENTAGE = 20 / 100
MIN_USED = 2
MAX_LEN = 10

class ColorDataset(data.Dataset):
    def __init__(self, vocab=None, split='Train', context_condition='far'):
        with open(os.path.join(RAW_DIR, 'filteredCorpus.csv')) as fp:
            df = pd.read_csv(fp)
        df = df[df['outcome'] == True]
        df = df[df['role'] == 'speaker']
        if context_condition != 'all':
            assert context_condition in ['far', 'close', 'split']
            df = df[df['condition'] == context_condition]

        self.texts = []
        self.rounds = []
        self.images = []
        self.textsList = [text for text in df['contents']]
        self.roundsList = [roundN for roundN in df['roundNum']]
        self.imagesList = list(zip([itemH for itemH in df['clickColH']], [itemS/100 for itemS in df['clickColS']], [itemL/100 for itemL in df['clickColL']]))
        
        length = len(self.textsList)
        train_len = int(length * TRAINING_PERCENTAGE)
        test_len = int(length * TESTING_PERCENTAGE)
        if split == 'Train':
            self.texts = self.textsList[:train_len]
            self.rounds = self.roundsList[:train_len]
            self.images = self.imagesList[:train_len]
        elif split == 'Validation':
            self.texts = self.textsList[train_len:-test_len]
            self.rounds = self.roundsList[train_len:-test_len]
            self.images = self.imagesList[train_len:-test_len]
        elif split == 'Test':
            self.texts = self.textsList[-test_len:]
            self.rounds = self.roundsList[-test_len:]
            self.images = self.imagesList[-test_len:]
        
        if vocab is None:
            self.vocab = self.build_vocab(self.texts)
        else:
            self.vocab = vocab
        self.vocab_size = len(self.vocab['w2i'])
        
        self.rgb_targets, self.texts = self.concatenate_by_round(self.texts, self.images, self.rounds)
        self.txt_sources, self.txt_targets, self.lengths, self.max_len = self.process_texts(self.texts)
        self.rgb_targets = np.array(self.rgb_targets)

        print("... split: {} dataset preparation complete.\n".format(split))

    def process_texts(self, texts):
        sources, targets, lengths = [], [], []

        n = len(texts)
        for i in range(n):
            tokens = preprocess_text(texts[i])
            source_tokens = [SOS_TOKEN] + tokens
            target_tokens = tokens + [EOS_TOKEN]
            assert len(source_tokens) == len(target_tokens)
            length = len(source_tokens)
            if length < MAX_LEN:
                source_tokens.extend([PAD_TOKEN] * (MAX_LEN - length))
                target_tokens.extend([PAD_TOKEN] * (MAX_LEN - length))
            else:
                source_tokens = source_tokens[:MAX_LEN]
                target_tokens = target_tokens[:MAX_LEN - 1] + [EOS_TOKEN]
                length = MAX_LEN
            assert len(source_tokens) == 10, breakpoint()
            assert len(target_tokens) == 10
            source_indices = [self.vocab['w2i'].get(token, self.vocab['w2i'][UNK_TOKEN]) for token in source_tokens]
            target_indices = [self.vocab['w2i'].get(token, self.vocab['w2i'][UNK_TOKEN]) for token in target_tokens]

            sources.append(np.array(source_indices))
            targets.append(np.array(target_indices))
            lengths.append(length)
        
        sources = np.array(sources)
        targets = np.array(targets)
        lengths = np.array(lengths)
        return sources, targets, lengths, MAX_LEN

    def concatenate_by_round(self, texts, images, rounds):
        concat_texts, rgb_targets = [], []
        concat = texts[0]
        for i in range(1, len(rounds)):
            if rounds[i] == rounds[i-1]:
                concat += " " + texts[i]
            else:
                rawRGB = np.array(hsl2rgb(images[i-1]))
                rgb_targets.append(rawRGB / 255.0)
                concat_texts.append(concat)
                concat = texts[i]
        return rgb_targets, concat_texts

    def build_vocab(self, texts):
        print("building vocab ...")
        w2c = defaultdict(int)
        i2w, w2i = {}, {}
        for text in texts:
            tokens = preprocess_text(text)
            for token in tokens:
                w2c[token] += 1
        indexCount = 0
        for token in w2c.keys():
            if w2c[token] >= MIN_USED:
                w2i[token] = indexCount
                i2w[indexCount] = token
                indexCount += 1
        w2i[SOS_TOKEN] = indexCount
        w2i[EOS_TOKEN] = indexCount+1
        w2i[UNK_TOKEN] = indexCount+2
        w2i[PAD_TOKEN] = indexCount+3
        i2w[indexCount] = SOS_TOKEN
        i2w[indexCount+1] = EOS_TOKEN
        i2w[indexCount+2] = UNK_TOKEN
        i2w[indexCount+3] = PAD_TOKEN

        vocab = {'i2w': i2w, 'w2i': w2i}
        print("==> total number of tokens: %d" % len(w2c.keys()))
        print("==> total number of tokens used at least twice (vocab_size): %d" % len(w2i))
        print("... vocab building done.")
        return vocab

    def __len__(self):
        return len(self.txt_sources)

    def __getitem__(self, index):
        return self.rgb_targets[index], self.txt_sources[index], self.txt_targets[index], self.lengths[index]

class WeakSup_ColorDataset(ColorDataset):
    def __init__(self, vocab=None, supervision_level=1.0, context_condition='far'):
        super(WeakSup_ColorDataset, self).__init__(vocab=vocab, split='Train', context_condition=context_condition)
        
        self.random_state = np.random.RandomState(18192)
        n = len(self.txt_sources)
        supervision = self.random_state.binomial(1, supervision_level, size=n)
        supervision = supervision.astype(np.bool)
        self.txt_sources = self.txt_sources[supervision]
        self.txt_targets = self.txt_targets[supervision]
        self.rgb_targets = self.rgb_targets[supervision]
        self.lengths = self.lengths[supervision]

class Colors_ReferenceGame(data.Dataset):
    def __init__(self, vocab, split='Test', context_condition='far'):
        with open(os.path.join(RAW_DIR, 'filteredCorpus.csv')) as fp:
            df = pd.read_csv(fp)
        # Only pick out data with true outcomes, far(=easy) conditions, and speaker text
        df = df[df['outcome'] == True]
        df = df[df['role'] == 'speaker']
        if context_condition != 'all':
            assert context_condition in ['far', 'close', 'split']
            df = df[df['condition'] == context_condition]
        
        self.texts = []
        self.rounds = []
        self.tgt_images = []
        self.d1_images = []
        self.d2_image = []

        # Convert csv dataframe into python lists
        self.textsList = [text for text in df['contents']]
        self.roundsList = [roundN for roundN in df['roundNum']]
        self.tgt_imagesList = list(zip([itemH for itemH in df['clickColH']], [itemS/100 for itemS in df['clickColS']], [itemL/100 for itemL in df['clickColL']]))
        self.d1_imagesList = list(zip([itemH for itemH in df['alt1ColH']], [itemS/100 for itemS in df['alt1ColS']], [itemL/100 for itemL in df['alt1ColL']]))
        self.d2_imagesList = list(zip([itemH for itemH in df['alt2ColH']], [itemS/100 for itemS in df['alt2ColS']], [itemL/100 for itemL in df['alt2ColL']]))

        length = len(self.textsList)
        train_len = int(length * TRAINING_PERCENTAGE)
        test_len = int(length * TESTING_PERCENTAGE)
        if split == 'Train':
            self.texts = self.textsList[:train_len]
            self.rounds = self.roundsList[:train_len]
            self.tgt_images = self.tgt_imagesList[:train_len]
            self.d1_images = self.d1_imagesList[:train_len]
            self.d2_images = self.d2_imagesList[:train_len]
        elif split == 'Validation':
            self.texts = self.textsList[train_len:-test_len]
            self.rounds = self.roundsList[train_len:-test_len]
            self.tgt_images = self.tgt_imagesList[train_len:-test_len]
            self.d1_images = self.d1_imagesList[train_len:-test_len]
            self.d2_images = self.d2_imagesList[train_len:-test_len]
        elif split == 'Test':
            self.texts = self.textsList[-test_len:]
            self.rounds = self.roundsList[-test_len:]
            self.tgt_images = self.tgt_imagesList[-test_len:]
            self.d1_images = self.d1_imagesList[-test_len:]
            self.d2_images = self.d2_imagesList[-test_len:]
        
        if vocab is None:
            self.vocab = self.build_vocab(self.texts)
        else:
            self.vocab = vocab

        self.vocab_size = len(self.vocab['w2i'])
        self.rgb_targets, self.d1_RGBs, self.d2_RGBs, self.texts = \
                self.concatenate_by_round(self.texts, self.tgt_images, self.d1_images, self.d2_images, self.rounds)
        self.txt_sources, self.txt_targets, self.lengths, self.max_len = self.process_texts(self.texts)
        
        self.rgb_targets = np.array(self.rgb_targets)
        self.d1_RGBs = np.array(self.d1_RGBs)
        self.d2_RGBs = np.array(self.d2_RGBs)

        print("... split: {} || dataset preparation complete.\n".format(split))

    def build_vocab(self, texts):
        print("building vocab ...")
        w2c = defaultdict(int)
        i2w, w2i = {}, {}
        for text in texts:
            tokens = preprocess_text(text)
            for token in tokens:
                w2c[token] += 1
        indexCount = 0
        for token in w2c.keys():
            if w2c[token] >= MIN_USED:
                w2i[token] = indexCount
                i2w[indexCount] = token
                indexCount += 1
        w2i[SOS_TOKEN] = indexCount
        w2i[EOS_TOKEN] = indexCount+1
        w2i[UNK_TOKEN] = indexCount+2
        w2i[PAD_TOKEN] = indexCount+3
        i2w[indexCount] = SOS_TOKEN
        i2w[indexCount+1] = EOS_TOKEN
        i2w[indexCount+2] = UNK_TOKEN
        i2w[indexCount+3] = PAD_TOKEN

        vocab = {'i2w': i2w, 'w2i': w2i}
        print("==> total number of tokens: %d" % len(w2c.keys()))
        print("==> total number of tokens used at least twice (vocab_size): %d" % len(w2i))
        print("... vocab building done.")
        return vocab

    def process_texts(self, texts):
        sources, targets, lengths = [], [], []

        n = len(texts)
        for i in range(n):
            tokens = preprocess_text(texts[i])
            source_tokens = [SOS_TOKEN] + tokens
            target_tokens = tokens + [EOS_TOKEN]
            assert len(source_tokens) == len(target_tokens)
            length = len(source_tokens)
            if length < MAX_LEN:
                source_tokens.extend([PAD_TOKEN] * (MAX_LEN - length))
                target_tokens.extend([PAD_TOKEN] * (MAX_LEN - length))
            else:
                source_tokens = source_tokens[:MAX_LEN]
                target_tokens = target_tokens[:MAX_LEN - 1] + [EOS_TOKEN]
                length = MAX_LEN
            assert len(source_tokens) == 10, breakpoint()
            assert len(target_tokens) == 10
            source_indices = [self.vocab['w2i'].get(token, self.vocab['w2i'][UNK_TOKEN]) for token in source_tokens]
            target_indices = [self.vocab['w2i'].get(token, self.vocab['w2i'][UNK_TOKEN]) for token in target_tokens]

            sources.append(np.array(source_indices))
            targets.append(np.array(target_indices))
            lengths.append(length)
        
        sources = np.array(sources)
        targets = np.array(targets)
        lengths = np.array(lengths)
        return sources, targets, lengths, MAX_LEN

    def concatenate_by_round(self, texts, tgt_images, d1_images, d2_images, rounds):
        concat_texts, rgb_targets, d1_RGBs, d2_RGBs = [], [], [], []
        concat = texts[0]
        for i in range(1, len(rounds)):
            if rounds[i] == rounds[i-1]:
                concat += " " + texts[i]
            else:
                tgt_rawRGB = np.array(hsl2rgb(tgt_images[i-1]))
                rgb_targets.append(tgt_rawRGB / 255.0)
                d1_rawRGB = np.array(hsl2rgb(d1_images[i-1]))
                d1_RGBs.append(d1_rawRGB / 255.0)
                d2_rawRGB = np.array(hsl2rgb(d2_images[i-1]))
                d2_RGBs.append(d2_rawRGB / 255.0)
                concat_texts.append(concat)
                concat = texts[i]
        return rgb_targets, d1_RGBs, d2_RGBs, concat_texts

    def __len__(self):
        return len(self.txt_sources)

    def __getitem__(self, index):
        return self.rgb_targets[index], self.d1_RGBs[index], self.d2_RGBs[index], self.txt_sources[index], self.txt_targets[index], self.lengths[index]

class WeakSup_ColorReference(Colors_ReferenceGame):
    def __init__(self, vocab=None, split='Train', supervision_level=1.0, context_condition='far'):
        super(WeakSup_ColorReference, self).__init__(vocab=vocab, split=split, context_condition=context_condition)
        
        self.random_state = np.random.RandomState(18192)
        n = len(self.txt_sources)
        supervision = self.random_state.binomial(1, supervision_level, size=n)
        supervision = supervision.astype(np.bool)
        self.txt_sources = self.txt_sources[supervision]
        self.txt_targets = self.txt_targets[supervision]
        self.rgb_targets = self.rgb_targets[supervision]
        self.d1_RGBs = self.d1_RGBs[supervision]
        self.d2_RGBs = self.d2_RGBs[supervision]
        self.lengths = self.lengths[supervision]

