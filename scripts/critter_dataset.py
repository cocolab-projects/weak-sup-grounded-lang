from __future__ import print_function

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from utils import OrderedCounter
from nltk import sent_tokenize, word_tokenize

import torch
import torch.utils.data as data
from torchvision import transforms
from collections import defaultdict

FILE_DIR = os.path.realpath(os.path.dirname(__file__))
RAW_DIR = os.path.join(FILE_DIR, '/mnt/fs5/hokysung/datasets/')
DATA_DIR = os.path.join(RAW_DIR, 'pilot_coll1')


SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
TRAINING_PERCENTAGE = 64 / 100
TESTING_PERCENTAGE = 20 / 100
MIN_USED = 1
MAX_LEN = 15

class Critters_ReferenceGame(data.Dataset):
    def __init__(self, vocab=None, split='Train', context_condition='all', 
                 image_size=32, image_transform=None, dataVal=None):
        super(Critters_ReferenceGame, self).__init__()

        self.split = split
       
        print('loading CSV')
        if (self.split == "Train"):
            csv_path = os.path.join(DATA_DIR, 'train/msgs.tsv')
            csv_path_concat = os.path.join(DATA_DIR, 'train/msgs_concat.tsv')
            csv_path_data = os.path.join(DATA_DIR, 'train/vision/dataset.tsv')

        if (self.split == "Validation"):
            csv_path = os.path.join(DATA_DIR, 'val/msgs.tsv')
            csv_path_concat = os.path.join(DATA_DIR, 'val/msgs_concat.tsv')
            csv_path_data = os.path.join(DATA_DIR, 'val/vision/dataset.tsv')

        if (self.split == "Test"):
            csv_path = os.path.join(DATA_DIR, 'test/msgs.tsv')
            csv_path_concat = os.path.join(DATA_DIR, 'test/msgs_concat.tsv')
            csv_path_data = os.path.join(DATA_DIR, 'test/vision/dataset.tsv')
        
        df = pd.read_csv(csv_path_data, sep='\t')
        # note that target_chair is always the chair
        # so label is always 3

        df = df.dropna()
        data = np.asarray(df)

        self.data = data
        text = [d[1] for d in data]
    
        if vocab is None:
            print('building vocab.')
            self.vocab = self.build_vocab(text)
        else:
            self.vocab = vocab
        
        self.w2i, self.i2w = self.vocab['w2i'], self.vocab['i2w']
        self.vocab_size = len(self.w2i)

        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN

        self.sos_index = self.w2i[self.sos_token]
        self.eos_index = self.w2i[self.eos_token]
        self.pad_index = self.w2i[self.pad_token]
        self.unk_index = self.w2i[self.unk_token]

        self.inputs, self.targets, self.lengths, self.max_length = self.process_texts(text)

        self.image_transform = image_transform

    def build_vocab(self, texts):
        w2c = defaultdict(int)
        i2w, w2i = {}, {}
        for text in texts:
            tokens = preprocess_text_critters(text)
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

        print("total number of words in vocab: %d" % len(w2i))
        print("total number of different words: %d" % len(w2c.keys()))

        return vocab


    def process_texts(self, texts):
        sources, targets, lengths = [], [], []

        n = len(texts)
        for i in range(n):
            tokens = preprocess_text_critters(texts[i])
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
            assert len(source_tokens) == MAX_LEN, breakpoint()
            assert len(target_tokens) == MAX_LEN
            source_indices = [self.vocab['w2i'].get(token, self.vocab['w2i'][UNK_TOKEN]) for token in source_tokens]
            target_indices = [self.vocab['w2i'].get(token, self.vocab['w2i'][UNK_TOKEN]) for token in target_tokens]

            sources.append(np.array(source_indices))
            targets.append(np.array(target_indices))
            lengths.append(length)
        
        sources = np.array(sources)
        targets = np.array(targets)
        lengths = np.array(lengths)
        return sources, targets, lengths, MAX_LEN

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        _, msg, distr1, distr2, target = self.data[index]
        # chair_target = chair_target + '.png'
        if self.split == "Train":
            for root, dirs, files in os.walk(os.path.join(DATA_DIR, 'train/vision/imgs')):
                if distr1 in files:
                    image_np_1 = os.path.join(root, distr1)
                if distr2 in files:
                    image_np_2 = os.path.join(root, distr2)
                if target in files:
                    image_np_tgt = os.path.join(root, target)
        if self.split == "Validation":
            for root, dirs, files in os.walk(os.path.join(DATA_DIR, 'val/vision/imgs')):
                if distr1 in files:
                    image_np_1 = os.path.join(root, distr1)
                if distr2 in files:
                    image_np_2 = os.path.join(root, distr2)
                if target in files:
                    image_np_tgt = os.path.join(root, target)
        if self.split == "Test":
            for root, dirs, files in os.walk(os.path.join(DATA_DIR, 'test/vision/imgs')):
                if distr1 in files:
                    image_np_1 = os.path.join(root, distr1)
                if distr2 in files:
                    image_np_2 = os.path.join(root, distr2)
                if target in files:
                    image_np_tgt = os.path.join(root, target)

        image_np_1_PIL = Image.open(image_np_1).convert('RGB')
        image_np_2_PIL = Image.open(image_np_2).convert('RGB')
        image_np_tgt_PIL = Image.open(image_np_tgt).convert('RGB')

        if self.image_transform is not None:
            image_np_1_PIL = self.image_transform(image_np_1_PIL)
            image_np_2_PIL = self.image_transform(image_np_2_PIL)
            image_np_tgt_PIL = self.image_transform(image_np_tgt_PIL)

        inputs = self.inputs[index]
        targets = self.targets[index]
        length = self.lengths[index]

        trans = transforms.ToTensor()

        return trans(image_np_tgt_PIL), trans(image_np_1_PIL), trans(image_np_2_PIL), inputs, targets, length

    def preprocess_text(text):
        text = text.lower() 
        tokens = word_tokenize(text)

        return tokens

class Weaksup_Critters_Reference(Critters_ReferenceGame):
    def __init__(self, vocab=None, transform=None, supervision_level=1.0, split='Train', context_condition='all'):
        super(Weaksup_Critters_Reference, self).__init__(
                        vocab=vocab, split=split, context_condition=context_condition, image_transform=transform)
        
        self.random_state = np.random.RandomState(18192)
        n = len(self.inputs)
        supervision = self.random_state.binomial(1, supervision_level, size=n)
        supervision = supervision.astype(np.bool)
        self.data = list(np.array(self.data)[supervision])
        self.inputs = self.inputs[supervision]
        self.targets = self.targets[supervision]
        self.lengths = self.lengths[supervision]

def preprocess_text_critters(text):
    text = text.lower() 
    tokens = word_tokenize(text)
    return tokens



