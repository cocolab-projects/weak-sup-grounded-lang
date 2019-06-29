from __future__ import print_function

import os
import json
import torch 
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import colorsys

# from utils import (hsl2rgb, preprocess_text)

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
RAW_DIR = os.path.join(FILE_DIR, '../')

SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
TRAINING_PERCENTAGE = 64 / 100
TESTING_PERCENTAGE = 20 / 100
MIN_USED = 2
MAX_LEN = 10

class ColorDataset(data.Dataset):
    def __init__(self, vocab=None, split='Train'):
        with open(os.path.join(RAW_DIR, 'filteredCorpus.csv')) as fp:
            df = pd.read_csv(fp)
        df = df[df['outcome'] == True]
        df = df[df['role'] == 'speaker']
        df = df[df['condition'] == 'far']
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
        self.target_RGBs, self.texts = self.concatenate_by_round(self.texts, self.images, self.rounds)
        self.inputs, self.lengths, self.max_len = self.process_texts(self.texts)

    def process_texts(self, texts):
        inputs, lengths = [], []

        n = len(texts)
        for i in range(n):
            tokens = preprocess_text(texts[i])
            input_tokens = [SOS_TOKEN] + tokens
            if len(input_tokens) > MAX_LEN-1:
                input_tokens = input_tokens[:MAX_LEN-1] + [EOS_TOKEN]
                length = MAX_LEN
            else:
                input_tokens += [EOS_TOKEN]
                length = len(input_tokens)
                input_tokens.extend([PAD_TOKEN] * (MAX_LEN - length))
            input_indices = [self.vocab['w2i'].get(token, self.vocab['w2i'][UNK_TOKEN]) for token in input_tokens]
            assert(len(input_indices) == MAX_LEN), breakpoint()
            inputs.append(np.array(input_indices))
            lengths.append(length)
        
        inputs = np.array(inputs)
        lengths = np.array(lengths)
        return inputs, lengths, MAX_LEN

    def concatenate_by_round(self, texts, images, rounds):
        concat_texts, target_RGBs = [], []
        concat = texts[0]
        for i in range(1, len(rounds)):
            if rounds[i] == rounds[i-1]:
                concat += " " + texts[i]
            else:
                rawRGB = np.array(hsl2rgb(images[i-1]))
                target_RGBs.append(rawRGB / 255.0)
                concat_texts.append(concat)
                concat = texts[i]
        return target_RGBs, concat_texts

    def build_vocab(self, texts):
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
        print("total number of words used at least twice: %d" % len(w2i))
        print("total number of different words: %d" % len(w2c.keys()))
        print("max number of word usage: %d" % max(w2c.values()))
        return vocab

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.target_RGBs[index], self.inputs[index], self.lengths[index]


class Colors_ReferenceGame(data.Dataset):
    def __init__(self, vocab, split='Test'):
        assert vocab is not None

        with open(os.path.join(RAW_DIR, 'filteredCorpus.csv')) as fp:
            df = pd.read_csv(fp)
        # Only pick out data with true outcomes, far(=easy) conditions, and speaker text
        df = df[df['outcome'] == True]
        df = df[df['role'] == 'speaker']
        df = df[df['condition'] == 'far']
        
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
            training_length
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
        self.tgt_RGBs, self.d1_RGBs, self.d2_RGBs, self.texts = \
                self.concatenate_by_round(self.texts, self.tgt_images, self.d1_images, self.d2_images, self.rounds)
        self.inputs, self.lengths, self.max_len = self.process_texts(self.texts)

    def process_texts(self, texts):
        inputs, lengths = [], []
        n = len(texts)
        for i in range(n):
            tokens = preprocess_text(texts[i])
            input_tokens = [SOS_TOKEN] + tokens
            if len(input_tokens) > MAX_LEN-1:
                input_tokens = input_tokens[:MAX_LEN-1] + [EOS_TOKEN]
                length = MAX_LEN
            else:
                input_tokens += [EOS_TOKEN]
                length = len(input_tokens)
                input_tokens.extend([PAD_TOKEN] * (MAX_LEN - length))
            input_indices = [self.vocab['w2i'].get(token, self.vocab['w2i'][UNK_TOKEN]) for token in input_tokens]
            assert(len(input_indices) == MAX_LEN), breakpoint()
            inputs.append(np.array(input_indices))
            lengths.append(length)
        
        inputs = np.array(inputs)
        lengths = np.array(lengths)
        return inputs, lengths, MAX_LEN

    def concatenate_by_round(self, texts, tgt_images, d1_images, d2_images, rounds):
        concat_texts, tgt_RGBs, d1_RGBs, d2_RGBs = [], [], [], []
        concat = texts[0]
        for i in range(1, len(rounds)):
            if rounds[i] == rounds[i-1]:
                concat += " " + texts[i]
            else:
                tgt_rawRGB = np.array(hsl2rgb(tgt_images[i-1]))
                tgt_RGBs.append(tgt_rawRGB / 255.0)
                d1_rawRGB = np.array(hsl2rgb(d1_images[i-1]))
                d1_RGBs.append(d1_rawRGB / 255.0)
                d2_rawRGB = np.array(hsl2rgb(d2_images[i-1]))
                d2_RGBs.append(d2_rawRGB / 255.0)
                concat_texts.append(concat)
                concat = texts[i]
        return tgt_RGBs, d1_RGBs, d2_RGBs, concat_texts

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.tgt_RGBs[index], self.d1_RGBs[index], self.d2_RGBs[index], self.inputs[index], self.lengths[index]


def hsl2rgb(hsl):
    """Convert HSL coordinates to RGB coordinates.
    https://www.rapidtables.com/convert/color/hsl-to-rgb.html
    @param hsl: np.array of size 3
                contains H, S, L coordinates
    @return rgb: (integer, integer, integer)
                RGB coordinate
    """
    H, S, L = hsl[0], hsl[1], hsl[2]
    assert (0 <= H <= 360) and (0 <= S <= 1) and (0 <= L <= 1)

    C = (1 - abs(2 * L - 1)) * S
    X = C * (1 - abs((H / 60.) % 2 - 1))
    m = L - C / 2.

    if H < 60:
        Rp, Gp, Bp = C, X, 0
    elif H < 120:
        Rp, Gp, Bp = X, C, 0
    elif H < 180:
        Rp, Gp, Bp = 0, C, X
    elif H < 240:
        Rp, Gp, Bp = 0, X, C
    elif H < 300:
        Rp, Gp, Bp = X, 0, C
    elif H < 360:
        Rp, Gp, Bp = C, 0, X

    R = int((Rp + m) * 255.)
    G = int((Gp + m) * 255.)
    B = int((Bp + m) * 255.)
    return (R, G, B)


def preprocess_text(text):
    text = text.lower() 
    tokens = word_tokenize(text)
    i = 0
    while i < len(tokens):
        while (tokens[i] != '.' and '.' in tokens[i]):
            tokens[i] = tokens[i].replace('.','')
        while (tokens[i] != '\'' and '\'' in tokens[i]):
            tokens[i] = tokens[i].replace('\'','')
        while('-' in tokens[i] or '/' in tokens[i]):
            if tokens[i] == '/' or tokens[i] == '-':
                tokens.pop(i)
                i -= 1
            if '/' in tokens[i]:
                split = tokens[i].split('/')
                tokens[i] = split[0]
                i += 1
                tokens.insert(i, split[1])
            if '-' in tokens[i]:
                split = tokens[i].split('-')                
                tokens[i] = split[0]
                i += 1
                tokens.insert(i, split[1])
            if tokens[i-1] == '/' or tokens[i-1] == '-':
                tokens.pop(i-1)
                i -= 1
            if '/' in tokens[i-1]:
                split = tokens[i-1].split('/')
                tokens[i-1] = split[0]
                i += 1
                tokens.insert(i-1, split[1])
            if '-' in tokens[i-1]:
                split = tokens[i-1].split('-')                
                tokens[i-1] = split[0]
                i += 1
                tokens.insert(i-1, split[1])
        if tokens[i].endswith('er'):
            tokens[i] = tokens[i][:-2]
            i += 1
            tokens.insert(i, 'er')
        if tokens[i].endswith('est'):
            tokens[i] = tokens[i][:-3]
            i += 1
            tokens.insert(i, 'est')
        if tokens[i].endswith('ish'):
            tokens[i] = tokens[i][:-3]
            i += 1
            tokens.insert(i, 'est')
        if tokens[i-1].endswith('er'):
            tokens[i-1] = tokens[i-1][:-2]
            i += 1
            tokens.insert(i-1, 'er')
        if tokens[i-1].endswith('est'):
            tokens[i-1] = tokens[i-1][:-3]
            i += 1
            tokens.insert(i-1, 'est')
        if tokens[i-1].endswith('ish'):
            tokens[i-1] = tokens[i-1][:-3]
            i += 1
            tokens.insert(i-1, 'est')
        i += 1
    replace = {'redd':'red', 'gren': 'green', 'whit':'white', 'biege':'beige', 'purp':'purple', 'olve':'olive', 'ca':'can', 'blu':'blue', 'orang':'orange', 'gray':'grey'}
    for i in range(len(tokens)):
        if tokens[i] in replace.keys():
            tokens[i] = replace[tokens[i]]
    while '' in tokens:
        tokens.remove('')
    return tokens
