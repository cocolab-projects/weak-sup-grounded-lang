from __future__ import print_function

import os
import json
import torch 
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import colorsys
import csv

from utils import (hsl2rgb, preprocess_text)

import nltk
from nltk import sent_tokenize, word_tokenize

import numpy as np
import torch
import torch.utils.data as data
from utils import hsl2rgb
from torchvision import transforms
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer

from collections import defaultdict, OrderedDict

FILE_DIR = os.path.realpath(os.path.dirname(__file__))
DATA_DIR = os.path.realpath(os.path.join(FILE_DIR, '/mnt/fs5/hokysung/ltprg/examples/games/csv/colorGrids/raw'))
OUT_DIR = os.path.realpath(os.path.join(FILE_DIR, '/mnt/fs5/hokysung/datasets/colorgrid_data'))

def process_single_file(state_filepath, utter_filepath, action_filepath):
	f = open(state_filepath, mode='rt')
	df = csv.DictReader(f, delimiter='\t')

	records = [record for record in df]
	for record in records:
		game_id = record['gameid']


	breakpoint()
	return

def process_single_game(file, game_id):
	assert utter_path.exists() and action_path.exists()

def merge_single_round(file, game_id, round_num):
	return

state_dir = os.path.join(DATA_DIR, '1/state')
utter_dir = os.path.join(DATA_DIR, '1/utterance')
action_dir = os.path.join(DATA_DIR, '1/action')

state_files = os.listdir(state_dir)
utter_files = os.listdir(utter_dir)
action_files = os.listdir(action_dir)

for state_filename in os.listdir(state_dir):
	print("== new iteration ==")
	action_filepath = os.path.join(action_dir, state_filename)
	if os.path.exists(action_filepath):
		for utter_filename in os.listdir(utter_dir):
			if state_filename[-16:] == utter_filename[-16:]:
				state_filepath = os.path.join(state_dir, state_filename)
				utter_filepath = os.path.join(utter_dir, utter_filename)
				process_single_file(state_filepath, utter_filepath, action_filepath)

