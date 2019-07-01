from __future__ import print_function

import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.nn.utils.rnn as rnn_utils

class Supervised(nn.Module):
    """
    @param embedding_module: nn.Embedding
                             pass the embedding module (share with
                             decoder)
    @param rgb_dim: final output should be rgb value, dimension 3
    @param hidden_dim: integer [default: 256]
                       number of hidden nodes in GRU
    """
    def __init__(self, vocab_size, rgb_dim=3, embedding_dim=64, hidden_dim=256):
        super(Supervised, self).__init__()
        assert (rgb_dim == 3)

        self.rgb_dim = rgb_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.sequential = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 4), \
                                        nn.ReLU(),  \
                                        nn.Linear(hidden_dim // 4, 3) )
    
    def forward(self, seq, length):
        batch_size = seq.size(0)

        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]

        # embed sequences
        embed_seq = self.embedding(seq)

        # pack padded sequences
        packed = rnn_utils.pack_padded_sequence(
            embed_seq,
            sorted_lengths.data.tolist() if batch_size > 1 else length.data.tolist(), batch_first=True)

        # forward RNN
        _, hidden = self.gru(packed)
        hidden = hidden[-1, ...]
        
        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]

        return self.sequential(hidden)

