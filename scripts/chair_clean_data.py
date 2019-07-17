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

def clean_data(data, names):
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

names = np.load(os.path.join(NUMPY_DIR, 'names.npy'))
images = np.load(os.path.join(NUMPY_DIR, 'images.npy'))
chair_list = []
for i in names:
    i = str(i.decode('utf-8'))
    chair_list.append(i)
names  = chair_list

print('loading CSV file ...')
csv_path = os.path.join(RAW_DIR, 'chairs2k_group_data.csv')
df = pd.read_csv(csv_path)
df = df[df['correct'] == True]
df = df[df['communication_role'] == 'speaker']

df = df[['chair_a', 'chair_b', 'chair_c', 'target_chair', 'text']]
df = df.dropna()
data = np.asarray(df)

# make sure rows reference existing images
data = clean_data(data, names)
np.save(os.path.join(RAW_DIR, "cleaned_data.npy"), data)
