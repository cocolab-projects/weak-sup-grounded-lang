from __future__ import print_function

import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.nn.utils.rnn as rnn_utils


class TextEmbedding(nn.Module):
    """ Embeds a |vocab_size| number

    """
    def __init__(self, vocab_size, hidden_dim=64):
        super(TextEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
    
    def forward(self, x):
        return self.embedding(x)


class Supervised(nn.Module):
    """Parameterizes q(z|sentence) where we use a recurrent
    model to project a sentence into a vector space.
    @param embedding_module: nn.Embedding
                             pass the embedding module (share with
                             decoder)
    @param d_dim: integer
                  number of latent dimensions
    @param hidden_dim: integer [default: 256]
                       number of hidden nodes in GRU
    """
    def __init__(self, embedding_module, rgb_dim=3, hidden_dim=256):
        super(Supervised, self).__init__()
        self.rgb_dim = rgb_dim
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding.embedding_dim
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

        # embed your sequences
        embed_seq = self.embedding(seq)

        # pack padded sequences
        packed = rnn_utils.pack_padded_sequence(
            embed_seq,
            sorted_lengths.data.tolist() if batch_size > 1 else length.data.tolist(), batch_first=True)

        # run RNN
        _, hidden = self.gru(packed)

        # remove layer number
        hidden = hidden[-1, ...]

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]

        return self.sequential(hidden)

