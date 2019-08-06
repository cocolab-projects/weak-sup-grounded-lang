from __future__ import print_function

import os
import json
from copy import deepcopy
from tqdm import tqdm
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
RAW_DIR = os.path.join(FILE_DIR, '/mnt/fs5/hokysung/datasets/chairs2k')
NUMPY_DIR = os.path.join(RAW_DIR, 'numpy')

SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
TRAINING_PERCENTAGE = 64 / 100
TESTING_PERCENTAGE = 20 / 100
MIN_USED = 2
MAX_LEN = 10

class Chairs_ReferenceGame(data.Dataset):
    def __init__(self, vocab=None, split='Train', context_condition='all', 
                 hard=False, image_size=32, image_transform=None):
        super(Chairs_ReferenceGame, self).__init__()

        self.images = np.load(os.path.join(NUMPY_DIR, 'images.npy'))
        self.context_condition = context_condition
        self.hard = hard
        self.split = split
       
        self.names = np.load(os.path.join(NUMPY_DIR, 'names.npy'))
        chair_list = []
        for i in self.names:
            i = str(i.decode('utf-8'))
            chair_list.append(i)
        self.names  = chair_list

        npy_path = os.path.join(RAW_DIR, 'cleaned_data.npy')
        if not os.path.exists(npy_path):
            # print('loading CSV file ...')
            csv_path = os.path.join(RAW_DIR, 'chairs2k_group_data.csv')
            df = pd.read_csv(csv_path)
            df = df[df['correct'] == True]
            df = df[df['communication_role'] == 'speaker']
            # note that target_chair is always the chair 
            # so label is always 3
            df = df[['chair_a', 'chair_b', 'chair_c', 'target_chair', 'text']]
            df = df.dropna()
            data = np.asarray(df)
            data = self.clean_data(data, self.names)
            np.save(npy_path, data)
        else:
            data = np.load(npy_path)

        target_names = data[:, 3]
        target_uniqs = np.unique(target_names)
        print('\nsplitting data into train and test -- condition "{}"'.format('hard' if self.hard else 'easy'))
        if not self.hard:
            # npy_path_easy = os.path.join(RAW_DIR, 'cleaned_data_easy.npy')
            # if not os.path.exists(npy_path_easy):
            # for each unique chair, divide all rows containing it into
            # training and test sets
            new_data = []
            pbar = tqdm(total=len(target_uniqs))
            for target in target_uniqs:
                data_i = data[target_names == target]
                train_len = int(TRAINING_PERCENTAGE * len(data_i))
                test_len = int(TESTING_PERCENTAGE * len(data_i))
                if (self.split == 'Train'):
                    new_data.append(data_i[:train_len])
                elif (self.split == 'Validation'):
                    new_data.append(data_i[train_len:-test_len])
                else:
                    new_data.append(data_i[-test_len:])
                pbar.update()
            pbar.close()
            new_data = np.concatenate(new_data, axis=0)
            # overwrite data variable
            data = new_data
            # np.save(npy_path_easy, data)
            # else:
            #     data = np.load(npy_path_easy)
        else:  # if difficulty is "hard", hard == True
            # npy_path_hard = os.path.join(RAW_DIR, 'cleaned_data_hard.npy')
            # if not os.path.exists(npy_path_hard):
            # for all chairs, divide into train and test sets
            train_len = int(TRAINING_PERCENTAGE * len(target_uniqs))
            test_len = int(TESTING_PERCENTAGE * len(target_uniqs))
            if (self.split == 'Train'):  
                splitter = np.in1d(target_names, target_uniqs[:train_len])
            elif (self.split == 'Validation'):  
                splitter = np.in1d(target_names, target_uniqs[train_len:-test_len])
            else:
                splitter = np.in1d(target_names, target_uniqs[-test_len:])
            data = data[splitter]
            # else:
            #     data = np.load(npy_path_hard)

        # replace target_chair with a label
        labels = []
        for i in range(len(data)):
            if data[i, 3] == data[i, 0]:
                labels.append(0)
            elif data[i, 3] == data[i, 1]:
                labels.append(1)
            elif data[i, 3] == data[i, 2]:
                labels.append(2)
            else:
                raise Exception('bad label')
        labels = np.array(labels)

        self.data = data
        self.labels = labels

        text = [d[-1] for d in data]
    
        if vocab is None:
            print('\nbuilding vocab ...')
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

        self.inputs, self.targets, self.lengths, self.positions, self.max_length \
            = self.process_texts(text)

        self.image_transform = image_transform

        print("{} dataset preparation complete.".format(split))

        # print(self.vocab)

    def build_vocab(self, texts):
        w2c = defaultdict(int)
        i2w, w2i = {}, {}
        for text in texts:
            tokens = preprocess_text_chairs(text)
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
        return vocab

    def clean_data(self, data, names):
        new_data = []
        for i in tqdm(range(len(data))):
            chair_a, chair_b, chair_c, chair_target, _ = data[i]
            if chair_a + '.png' not in names:
                continue
            if chair_b + '.png' not in names:
                continue
            if chair_c + '.png' not in names:
                continue
            if chair_target + '.png' not in names:
                continue
            new_data.append(data[i])
        new_data = np.array(new_data)
        return new_data

    def process_texts(self, texts):
        inputs, targets, lengths, positions = [], [], [], []

        n = len(texts)
        max_len = 0
        for i in range(n):
            text = texts[i]
            tokens = word_tokenize(text)
            input_tokens = [SOS_TOKEN] + tokens
            target_tokens = tokens + [EOS_TOKEN]
            assert len(input_tokens) == len(target_tokens)
            length = len(input_tokens)
            max_len = max(max_len, length)

            inputs.append(input_tokens)
            targets.append(target_tokens)
            lengths.append(length)

        for i in range(n):
            input_tokens = inputs[i]
            target_tokens = targets[i]
            length = lengths[i]
            input_tokens.extend([PAD_TOKEN] * (max_len - length))
            target_tokens.extend([PAD_TOKEN] * (max_len - length))
            input_tokens = [self.w2i.get(token, self.w2i[UNK_TOKEN]) for token in input_tokens]
            target_tokens = [self.w2i.get(token, self.w2i[UNK_TOKEN]) for token in target_tokens]
            pos = [pos_i+1 if w_i != self.pad_index else 0
                   for pos_i, w_i in enumerate(input_tokens)]
            inputs[i] = input_tokens
            targets[i] = target_tokens
            positions.append(pos)
        
        inputs = np.array(inputs)
        targets = np.array(targets)
        lengths = np.array(lengths)
        positions = np.array(positions)

        return inputs, targets, lengths, positions, max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        chair_a, chair_b, chair_c, chair_target, _ = self.data[index]
        label = self.labels[index]
        
        chair_a = chair_a + '.png'
        chair_b = chair_b + '.png'
        chair_c = chair_c + '.png'
        # chair_target = chair_target + '.png'

        chair_names = list(self.names)
        index_a = chair_names.index(chair_a)
        index_b = chair_names.index(chair_b)
        index_c = chair_names.index(chair_c)
        # index_target = chair_names.index(chair_target)

        chair_a_np = self.images[index_a][0]
        chair_b_np = self.images[index_b][0]
        chair_c_np = self.images[index_c][0]
        # chair_target_np = self.images[index_target][0]

        chair_a_pt = torch.from_numpy(chair_a_np).unsqueeze(0)
        chair_a = transforms.ToPILImage()(chair_a_pt).convert('RGB')

        chair_b_pt = torch.from_numpy(chair_b_np).unsqueeze(0)
        chair_b = transforms.ToPILImage()(chair_b_pt).convert('RGB')

        chair_c_pt = torch.from_numpy(chair_c_np).unsqueeze(0)
        chair_c = transforms.ToPILImage()(chair_c_pt).convert('RGB')

        if self.image_transform is not None:
            chair_a = self.image_transform(chair_a)
            chair_b = self.image_transform(chair_b)
            chair_c = self.image_transform(chair_c)

        inputs = self.inputs[index]
        targets = self.targets[index]
        length = self.lengths[index]
        trans = transforms.ToTensor()

        inputs = torch.from_numpy(inputs).long()
        targets = torch.from_numpy(targets).long()

        return trans(chair_a), trans(chair_b), trans(chair_c), inputs, targets, length

class Weaksup_Chairs_Reference(Chairs_ReferenceGame):
    def __init__(self, vocab=None, transform=None, supervision_level=1.0, split='Train', context_condition='all'):
        super(Weaksup_Chairs_Reference, self).__init__(
                        vocab=vocab, split=split, context_condition='all', image_transform=transform)
        
        self.random_state = np.random.RandomState(18192)
        n = len(self.inputs)
        supervision = self.random_state.binomial(1, supervision_level, size=n)
        supervision = supervision.astype(np.bool)
        self.data = list(np.array(self.data)[supervision])
        self.inputs = self.inputs[supervision]
        self.targets = self.targets[supervision]
        self.lengths = self.lengths[supervision]
        self.positions = self.positions[supervision]

def preprocess_text_chairs(text):
    text = text.lower() 
    tokens = word_tokenize(text)
    # i = 0
    # while i < len(tokens):
    #     while (tokens[i] != '.' and '.' in tokens[i]):
    #         tokens[i] = tokens[i].replace('.','')
    #     while (tokens[i] != '\'' and '\'' in tokens[i]):
    #         tokens[i] = tokens[i].replace('\'','')
    #     while('-' in tokens[i] or '/' in tokens[i]):
    #         if tokens[i] == '/' or tokens[i] == '-':
    #             tokens.pop(i)
    #             i -= 1
    #         if '/' in tokens[i]:
    #             split = tokens[i].split('/')
    #             tokens[i] = split[0]
    #             i += 1
    #             tokens.insert(i, split[1])
    #         if '-' in tokens[i]:
    #             split = tokens[i].split('-')                
    #             tokens[i] = split[0]
    #             i += 1
    #             tokens.insert(i, split[1])
    #         if tokens[i-1] == '/' or tokens[i-1] == '-':
    #             tokens.pop(i-1)
    #             i -= 1
    #         if '/' in tokens[i-1]:
    #             split = tokens[i-1].split('/')
    #             tokens[i-1] = split[0]
    #             i += 1
    #             tokens.insert(i-1, split[1])
    #         if '-' in tokens[i-1]:
    #             split = tokens[i-1].split('-')                
    #             tokens[i-1] = split[0]
    #             i += 1
    #             tokens.insert(i-1, split[1])
    #     if tokens[i].endswith('er'):
    #         tokens[i] = tokens[i][:-2]
    #         i += 1
    #         tokens.insert(i, 'er')
    #     if tokens[i].endswith('est'):
    #         tokens[i] = tokens[i][:-3]
    #         i += 1
    #         tokens.insert(i, 'est')
    #     if tokens[i].endswith('ish'):
    #         tokens[i] = tokens[i][:-3]
    #         i += 1
    #         tokens.insert(i, 'est')
    #     if tokens[i-1].endswith('er'):
    #         tokens[i-1] = tokens[i-1][:-2]
    #         i += 1
    #         tokens.insert(i-1, 'er')
    #     if tokens[i-1].endswith('est'):
    #         tokens[i-1] = tokens[i-1][:-3]
    #         i += 1
    #         tokens.insert(i-1, 'est')
    #     if tokens[i-1].endswith('ish'):
    #         tokens[i-1] = tokens[i-1][:-3]
    #         i += 1
    #         tokens.insert(i-1, 'est')
    #     i += 1
    while '' in tokens:
        tokens.remove('')
    return tokens



