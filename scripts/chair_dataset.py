from __future__ import print_function

import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from src.utils.utils import OrderedCounter
from nltk import sent_tokenize, word_tokenize

import torch
import torch.utils.data as data
from torchvision import transforms

FILE_DIR = os.path.realpath(os.path.dirname(__file__))
RAW_DIR = os.path.join(FILE_DIR, 'chair_data')
NUMPY_DIR = '/mnt/fs5/wumike/datasets/chairs2k/numpy'

SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'


def load_char_id_to_utterance_map():
    with open(os.path.join(RAW_DIR, 'chairs2k_group_data.csv')) as fp:
        df = pd.read_csv(fp)

    df = df[df['correct']]
    chair_id = np.asarray(df['selected_chair'])
    text = np.asarray(df['text'])

    return chair_id, text


class Chairs2k_ReferenceGame(data.Dataset):
    def __init__(self, data_dir, vocab, image_transform=None):
        assert vocab is not None

        self.names = np.load(os.path.join(NUMPY_DIR, 'names.npy'))
        self.images = np.load(os.path.join(NUMPY_DIR, 'images.npy'))
        
        npy_path = os.path.join(RAW_DIR, 'chairs2k_group_data.npy')
        if not os.path.exists(npy_path):
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

        self.data = data

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

        text = [d[-1] for d in data]
        self.inputs, self.targets, self.lengths, self.positions, self.max_length \
            = self.process_texts(text)

        self.image_transform = image_transform

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
        
        chair_a = chair_a + '.png'
        chair_b = chair_b + '.png'
        chair_c = chair_c + '.png'
        chair_target = chair_target + '.png'

        chair_names = list(self.names)
        index_a = chair_names.index(chair_a)
        index_b = chair_names.index(chair_b)
        index_c = chair_names.index(chair_c)
        index_target = chair_names.index(chair_target)

        chair_a_np = self.images[index_a][0]
        chair_b_np = self.images[index_b][0]
        chair_c_np = self.images[index_c][0]
        chair_target_np = self.images[index_target][0]

        chair_a_pt = torch.from_numpy(chair_a_np)
        chair_a = transforms.ToPILImage()(chair_a_pt).convert('RGB')

        chair_b_pt = torch.from_numpy(chair_b_np)
        chair_b = transforms.ToPILImage()(chair_b_pt).convert('RGB')

        chair_c_pt = torch.from_numpy(chair_c_np)
        chair_c = transforms.ToPILImage()(chair_c_pt).convert('RGB')

        chair_target_pt = torch.from_numpy(chair_target_np)
        chair_target = transforms.ToPILImage()(chair_target_pt).convert('RGB')

        if self.image_transform is not None:
            chair_a = self.image_transform(chair_a)
            chair_b = self.image_transform(chair_b)
            chair_c = self.image_transform(chair_c)
            chair_target = self.image_transform(chair_target)

        inputs = self.inputs[index]
        targets = self.targets[index]
        length = self.lengths[index]

        inputs = torch.from_numpy(inputs).long()
        targets = torch.from_numpy(targets).long()

        return chair_a, chair_b, chair_c, chair_target, inputs, targets, length


class Chairs2k_Numpy(data.Dataset):
    def __init__(self, data_dir, vocab=None, train=True, image_transform=None):
        self.names = np.load(os.path.join(NUMPY_DIR, 'names.npy'))
        self.images = np.load(os.path.join(NUMPY_DIR, 'images.npy'))
        self.names = np.array([os.path.splitext(name)[0] for name in self.names])

        _chair_id, _text = load_char_id_to_utterance_map()
        chair_id, text = [], []
        print('Subsetting the chair ID.')
        pbar = tqdm(total=len(_chair_id))
        for c, t in zip(_chair_id, _text):
            if c in self.names:
                chair_id.append(c)
                text.append(t)
            pbar.update()
        pbar.close()
        chair_id = np.array(chair_id)
        text = np.array(text)

        n_total = len(chair_id)
        n_train = int(0.8 * n_total)
        train_chair_id, test_chair_id = chair_id[:n_train], chair_id[n_train:]
        train_text, test_text = text[:n_train], text[n_train:]
        if train:
            chair_id = train_chair_id
            text = train_text
        else:
            chair_id = test_chair_id
            text = test_text

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

        print('processing text.')
        self.inputs, self.targets, self.lengths, self.positions, self.max_length \
            = self.process_texts(text)
        self.chair_id = chair_id

        self.image_transform = image_transform

    def process_texts(self, texts):
        inputs, targets, lengths, positions = [], [], [], []

        n = len(texts)
        max_len = 0
        for i in tqdm(range(n)):
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

    def build_vocab(self, texts):
        w2i = dict()
        i2w = dict()
        w2c = OrderedCounter()
        special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)
        
        pbar = tqdm(total=len(texts))
        for text in texts:
            tokens = word_tokenize(text)
            w2c.update(tokens)
            pbar.update()
        pbar.close()

        for w, c in w2c.items():
            i2w[len(w2i)] = w
            w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)
        vocab = dict(w2i=w2i, i2w=i2w)

        return vocab

    def __len__(self):
        return len(self.chair_id)

    def __getitem__(self, index):
        chair_id = self.chair_id[index]
        ix = np.where(self.names == chair_id)[0]
        image_np = self.images[ix][0] 
        image_pt = torch.from_numpy(image_np)
        image = transforms.ToPILImage()(image_pt)
    
        if self.image_transform is not None:
            image = self.image_transform(image)

        inputs = self.inputs[index]
        targets = self.targets[index]
        length = self.lengths[index]
        pos = self.positions[index]

        inputs = torch.from_numpy(inputs).long()
        targets = torch.from_numpy(targets).long()
        pos = torch.from_numpy(pos).long()

        return image, inputs, targets, length, pos


class WeakSup_Chairs2k_Numpy(Chairs2k_Numpy):
    def __init__(self, data_dir, vocab=None, transform=None, supervision_level=1.0):
        super(WeakSup_Chairs2k_Numpy, self).__init__(
            data_dir, vocab=vocab, train=True, image_transform=transform)
        
        self.random_state = np.random.RandomState(18192)
        n = len(self.inputs)
        supervision = self.random_state.binomial(1, supervision_level, size=n)
        supervision = supervision.astype(np.bool)
        self.chair_id = list(np.array(self.chair_id)[supervision])
        self.inputs = self.inputs[supervision]
        self.targets = self.targets[supervision]
        self.lengths = self.lengths[supervision]
        self.positions = self.positions[supervision]