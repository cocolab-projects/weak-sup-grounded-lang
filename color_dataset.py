from __future__ import print_function

import os
import json
import torch 
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import colorsys
# from src.utils.utils import OrderedCounter
from nltk import sent_tokenize, word_tokenize
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict

FILE_DIR = os.path.realpath(os.path.dirname(__file__))
RAW_DIR = os.path.join(FILE_DIR, 'data')
print(RAW_DIR)

SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
TRAINING_PERCENTAGE = 64 / 100
TESTING_PERCENTAGE = 20 / 100
MIN_USED = 2
MAX_LEN = 10

# Name for now
class ColorDataset(data.Dataset):
# init?? self
    def __init__(self, vocab=None, train=True):
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
        if train:
            self.texts = self.textsList[:int(len(self.textsList) * TRAINING_PERCENTAGE)]
            self.rounds = self.roundsList[:int(len(self.roundsList) * TRAINING_PERCENTAGE)]
            self.images = self.imagesList[:int(len(self.imagesList) * TRAINING_PERCENTAGE)]
        else:
            self.texts = self.textsList[-int(len(self.textsList) * TESTING_PERCENTAGE):]
            self.rounds = self.roundsList[-int(len(self.roundsList) * TESTING_PERCENTAGE):]
            self.images = self.imagesList[-int(len(self.imagesList) * TESTING_PERCENTAGE):]
        if vocab is None:
            self.vocab = self.build_vocab(self.texts)
        else:
            self.vocab = vocab
        self.target_images, self.texts = self.concatenate_by_round(self.texts, self.images, self.rounds)
        self.inputs, self.lengths, self.max_len = self.process_texts(self.texts)
        print(self.vocab['i2w'])
        print(self.target_images[8])
        print(self.inputs[8])

    def process_texts(self, texts):
        print("process_texts")
        inputs, lengths = [], []

        n = len(texts)
        max_len = 0
        for i in range(n):
            tokens = self.preprocess_text(texts[i])
            input_tokens = [SOS_TOKEN] + tokens + [EOS_TOKEN]
            length = len(input_tokens)
            if length == 121:
                print(texts[i])

            inputs.append(input_tokens)
            lengths.append(length)

        for i in range(n):
            input_tokens = inputs[i]
            length = lengths[i]
            input_tokens.extend([PAD_TOKEN] * (MAX_LEN - length))
            input_tokens = [self.vocab['w2i'].get(token, self.vocab['w2i'][UNK_TOKEN]) for token in input_tokens]
            # pos = [pos_i+1 if w_i != w2i[PAD_TOKEN] else 0
            #         for pos_i, w_i in enumerate(input_tokens)]
            inputs[i] = input_tokens
            # positions.append(pos)
        
        inputs = np.array(inputs)
        lengths = np.array(lengths)
        # positions = np.array(positions)

        print(inputs)
        print(lengths)
        print(max_len)
        return inputs, lengths, max_len

    def concatenate_by_round(self, texts, images, rounds):
        concat_texts, target_images = [], []
        concat = texts[0]
        for i in range(1, len(rounds)):
            if rounds[i] == rounds[i-1]:
                concat += " " + texts[i]
            else:
                target_images.append(self.hsl2rgb(images[i-1]))
                concat_texts.append(concat)
                concat = texts[i]
        return target_images, concat_texts

    def build_vocab(self, texts):
        w2c = defaultdict(int)
        i2w, w2i = {}, {}
        for text in texts:
            tokens = self.preprocess_text(text)
            for token in tokens:
                w2c[token] += 1
        indexCount = 0
        for token in w2c.keys():
            if w2c[token] >= MIN_USED:
                w2i[token] = indexCount
                i2w[indexCount] = token
                indexCount += 1
        w2i[SOS_TOKEN] = indexCount+1
        w2i[EOS_TOKEN] = indexCount+2
        w2i[UNK_TOKEN] = indexCount+3
        w2i[PAD_TOKEN] = indexCount+4
        i2w[indexCount+1] = SOS_TOKEN
        i2w[indexCount+2] = EOS_TOKEN
        i2w[indexCount+3] = UNK_TOKEN
        i2w[indexCount+4] = PAD_TOKEN

        vocab = {'i2w': i2w, 'w2i': w2i}
        print("total number of words used at least twice: %d" % len(w2i))
        print("total number of different words: %d" % len(w2c.keys()))
        print("max number of word usage: %d" % max(w2c.values()))
        return vocab

    def preprocess_text(self, text):
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

    def hsl2rgb(self, hsl):
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
    def __len__(self):
        return len(self.setUsed)

    def __getitem__(self, index):

        return image, self.inputs[index], self.lengths[index]