from __future__ import print_function

import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.nn.utils.rnn as rnn_utils

class TextImageCompatibility(nn.Module):
    def __init__(self, vocab_size, img_size=32, channels=3, embedding_dim=64, hidden_dim=256, n_filters=64):
        super(TextImageCompatibility, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim

        self.hidden_dim = 256

        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.txt_lin = nn.Linear(self.hidden_dim, self.hidden_dim // 2)

        self.conv = nn.Sequential(
            nn.Conv2d(channels, n_filters, 2, 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters * 2, 2, 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_filters * 2, n_filters * 4, 2, 2, padding=0))
        cout = gen_32_conv_output_dim(img_size)
        self.fc = nn.Linear(n_filters * 4 * cout**2, hidden_dim // 2)
        self.cout = cout
        self.n_filters = n_filters
        self.sequential = nn.Sequential(
                                        nn.Linear(self.hidden_dim, self.hidden_dim // 3), \
                                        nn.ReLU(),  \
                                        nn.Linear(self.hidden_dim // 3, self.hidden_dim // 9), \
                                        nn.ReLU(), \
                                        nn.Linear(self.hidden_dim // 9, self.hidden_dim // 27), \
                                        nn.ReLU(), \
                                        nn.Linear(self.hidden_dim // 27, 1))

    def forward(self, img, seq, length):
        assert img.size(0) == seq.size(0)
        batch_size = img.size(0)

        # CNN portion for image
        out = self.conv(img)
        out = out.view(batch_size, self.n_filters * 4 * self.cout**2)
        img_hidden = self.fc(out)

        # RNN portion for text
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
        txt_hidden = self.txt_lin(hidden)

        # concat then forward
        concat = torch.cat((txt_hidden, img_hidden), 1)
        return self.sequential(concat)

def gen_32_conv_output_dim(s):
    s = get_conv_output_dim(s, 2, 0, 2)
    s = get_conv_output_dim(s, 2, 0, 2)
    s = get_conv_output_dim(s, 2, 0, 2)
    return s

def get_conv_output_dim(I, K, P, S):
    # I = input height/length
    # K = filter size
    # P = padding
    # S = stride
    # O = output height/length
    O = (I - K + 2*P)/float(S) + 1
    return int(O)

class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256):
        super(TextEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
    
    def forward(self, x):
        return self.embedding(x)

class TextEncoder(nn.Module):
    """
    x: text, y: image, z: latent
    Model p(z|x)
    @param embedding_module: nn.Embedding
                             pass the embedding module (share with
                             decoder)
    @param z_dim: number of latent dimensions
    @param hidden_dim: integer [default: 256]
                       number of hidden nodes in GRU
    """
    def __init__(self, embedding_module, z_dim, hidden_dim=256):
        super(TextEncoder, self).__init__()
        self.z_dim = z_dim
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding.embedding_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, z_dim * 2)
    
    def forward(self, seq, lengths):
        batch_size = seq.size(0)

        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
            seq = seq[sorted_idx]

        # embed sequences
        embed_seq = self.embedding(seq)

        # pack padded sequences
        packed = rnn_utils.pack_padded_sequence(
            embed_seq,
            sorted_lengths.data.tolist() if batch_size > 1 else lengths.data.tolist(), batch_first=True)

        # forward RNN
        _, hidden = self.gru(packed)
        hidden = hidden[-1, ...]
        
        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]

        z_mu, z_logvar = torch.chunk(self.linear(hidden), 2, dim=1)

        return z_mu, z_logvar

class TextDecoder(nn.Module):
    """
    x: text, y: image, z: latent
    Model p(x|z)
    @param embedding_module: nn.Embedding
                             pass the embedding module (share with decoder)
    @param z_dim: number of latent dimensions
    @param hidden_dim: integer [default: 256]
                       number of hidden nodes in GRU
    """
    def __init__(self, embedding_module, z_dim, sos_index, eos_index,
                 pad_index, unk_index, hidden_dim=256, word_dropout=0.,
                 embedding_dropout=0.):
        super(TextDecoder, self).__init__()
        self.z_dim = z_dim
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding.embedding_dim
        self.vocab_size = embedding_module.embedding.num_embeddings
        self.hidden_dim = hidden_dim
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.outputs2vocab = nn.Linear(self.hidden_dim, self.vocab_size)
        self.latent2hidden = nn.Linear(self.z_dim, self.hidden_dim)
        self.word_dropout = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
    
    def forward(self, z_sample, seq, length):
        batch_size = seq.size(0)

        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            z_sample = z_sample[sorted_idx]
            seq = seq[sorted_idx]
        else:
            sorted_lengths = length  # since we use this variable later

        if self.word_dropout > 0:
            # randomly replace with unknown tokens
            prob = torch.rand(seq.size())
            prob[(seq.cpu().data - self.sos_index) & \
                 (seq.cpu().data - self.pad_index) == 0] = 1
            mask_seq = seq.clone()
            mask_seq[(prob < self.word_dropout).to(z_sample.device)] = self.unk_index
            seq = mask_seq

        # embed sequences
        embed_seq = self.embedding(seq)

        # pack padded sequences
        packed = rnn_utils.pack_padded_sequence(embed_seq, sorted_lengths, batch_first=True)

        # initialize hidden (initialize |z| part in |p(x_i|z, x_{i-1})| )
        hidden = self.latent2hidden(z_sample)
        hidden = hidden.unsqueeze(0).contiguous()

        # forward RNN (recurrently obtain |x_i| given |z| and |x_{i-1})
        packed_output, _ = self.gru(packed, hidden)
        output = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
        output = output[0].contiguous()
        
        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            output = output[reversed_idx]

        max_length = output.size(1)
        output_2d = output.view(batch_size * max_length, self.hidden_dim)
        outputs_2d = self.outputs2vocab(output_2d)
        outputs = outputs_2d.view(batch_size, max_length, self.vocab_size)

        return outputs


class ImageEncoder(nn.Module):
    def __init__(self, channels, img_size, z_dim, n_filters=64):
        super(ImageEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, n_filters, 2, 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters * 2, 2, 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_filters * 2, n_filters * 4, 2, 2, padding=0),
            nn.ReLU())
        cout = gen_32_conv_output_dim(img_size)
        self.fc = nn.Linear(n_filters * 4 * cout**2, z_dim * 2)
        self.cout = cout
        self.n_filters = n_filters

    def forward(self, img):
        batch_size = img.size(0)
        out = self.conv(img)
        out = out.view(batch_size, self.n_filters * 4 * self.cout**2)
        z_params = self.fc(out)
        z_mu, z_logvar = torch.chunk(z_params, 2, dim=1)

        return z_mu, z_logvar

class ImageDecoder(nn.Module):
    def __init__(self, channels, img_size, z_dim, n_filters=64):
        super(ImageDecoder, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(n_filters * 4, n_filters * 4, 2, 2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 2, 2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filters * 2, n_filters, 2, 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_filters, channels, 1, 1, padding=0))
        cout = gen_32_conv_output_dim(img_size)
        self.fc = nn.Sequential(
            nn.Linear(z_dim, n_filters * 4 * cout**2),
            nn.ReLU())
        self.cout = cout
        self.n_filters = n_filters
        self.channels = channels
        self.img_size = img_size

    def forward(self, z):
        if z.dim() != 1:
            batch_size = z.size(0)
        else:
            batch_size = 1
        out = self.fc(z)
        out = out.view(batch_size, self.n_filters * 4, self.cout, self.cout)
        out = self.conv(out)
        x_logits = out.view(batch_size, self.channels, self.img_size, self.img_size)
        x_mu = torch.sigmoid(x_logits)

        return x_mu

class ImageTextEncoder(nn.Module):
    def __init__(self, channels, img_size, z_dim, embedding_module, text_hidden_dim=256, n_filters=64):
        super(ImageTextEncoder, self).__init__()
        self.text_embedding = embedding_module
        self.embedding_dim = embedding_module.embedding.embedding_dim
        self.text_model = nn.GRU(self.embedding_dim, text_hidden_dim)
        self.image_model = nn.Sequential(
            nn.Conv2d(channels, n_filters, 2, 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters * 2, 2, 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_filters * 2, n_filters * 4, 2, 2, padding=0),
            nn.ReLU())
        cout = gen_32_conv_output_dim(img_size)
        self.fc = nn.Linear(n_filters * 4 * cout**2 + text_hidden_dim, z_dim * 2)
        self.cout = cout
        self.n_filters = n_filters

    def text_forward(self, seq, length):
        batch_size = seq.size(0)

        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]

        # embed your sequences
        embed_seq = self.text_embedding(seq)

        # reorder from (B,L,D) to (L,B,D)
        embed_seq = embed_seq.transpose(0, 1)

        packed = rnn_utils.pack_padded_sequence(
            embed_seq,
            sorted_lengths.data.tolist() if batch_size > 1 else length.data.tolist())

        _, hidden = self.text_model(packed)
        hidden = hidden[-1, ...]

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]

        return hidden

    def forward(self, img, text, length):
        batch_size = img.size(0)
        out_img = self.image_model(img)
        out_img = out_img.view(batch_size, self.n_filters * 4 * self.cout**2)
        out_txt = self.text_forward(text, length)
        out = torch.cat((out_img, out_txt), dim=1)
        z_params = self.fc(out)
        z_mu, z_logvar = torch.chunk(z_params, 2, dim=1)

        return z_mu, z_logvar


########## Models for Colors dataset only (RGB & text) ##########

class ColorSupervised(nn.Module):
    """
    Supervised, x: text, y: image (rgb value)
    @param embedding_module: nn.Embedding
                             pass the embedding module (share with
                             decoder)
    @param rgb_dim: final output should be rgb value, dimension 3
    @param hidden_dim: integer [default: 256]
                       number of hidden nodes in GRU
    """
    def __init__(self, vocab_size, rgb_dim=3, embedding_dim=64, hidden_dim=256):
        super(ColorSupervised, self).__init__()
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

        return torch.sigmoid(self.sequential(hidden))

class ColorSupervised_Paired(nn.Module):
    """
    Supervised, x: text, y: image (rgb value)
    @param embedding_module: nn.Embedding
                             pass the embedding module (share with
                             decoder)
    @param rgb_dim: final output should be rgb value, dimension 3
    @param hidden_dim: integer [default: 256]
                       number of hidden nodes in GRU
    """
    def __init__(self, vocab_size, rgb_dim=3, embedding_dim=64, hidden_dim=256):
        super(ColorSupervised_Paired, self).__init__()
        assert (rgb_dim == 3)

        self.rgb_dim = rgb_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.txt_lin = nn.Linear(hidden_dim, hidden_dim // 2)
        self.rgb_seq = nn.Sequential(nn.Linear(rgb_dim, hidden_dim), \
                                        nn.ReLU(),  \
                                        nn.Linear(hidden_dim, hidden_dim // 2))
        self.sequential = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), \
                                        nn.ReLU(),  \
                                        nn.Linear(hidden_dim // 2, 1))
    
    def forward(self, rgb, seq, length):
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

        txt_hidden = self.txt_lin(hidden)
        rgb_hidden = self.rgb_seq(rgb)

        concat = torch.cat((txt_hidden, rgb_hidden), 1)

        return self.sequential(concat)

class ColorEncoder(nn.Module):
    """
    x: text, y: image, z: latent
    Model p(z|y)
    @param z_dim: number of latent dimensions
    @param hidden_dim: integer [default: 256]
                       number of hidden nodes in GRU
    """
    def __init__(self, z_dim, rgb_dim=3, hidden_dim=256):
        super(ColorEncoder, self).__init__()
        assert (rgb_dim == 3)

        self.rgb_dim = rgb_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.sequential = nn.Sequential(nn.Linear(rgb_dim, hidden_dim), \
                                        nn.ReLU(),  \
                                        nn.Linear(hidden_dim, z_dim * 2))
    
    def forward(self, rgb):
        # sent rgb value to latent dimension
        z_mu, z_logvar = torch.chunk(self.sequential(rgb), 2, dim=1)
        
        return z_mu, z_logvar

class ColorEncoder_Augmented(nn.Module):
    """
    x: text, y: image, z: latent
    Model p(z|y)
    @param z_dim: number of latent dimensions
    @param hidden_dim: integer [default: 256]
                       number of hidden nodes in GRU
    """
    def __init__(self, z_dim, rgb_dim=3, hidden_dim=256):
        super(ColorEncoder_Augmented, self).__init__()
        assert (rgb_dim == 3)

        self.rgb_dim = rgb_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.sequential = nn.Sequential(nn.Linear(rgb_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim * 4),
                                        nn.LeakyReLU(),
                                        nn.Linear(hidden_dim * 4, hidden_dim * 3),
                                        nn.LeakyReLU(),
                                        nn.Linear(hidden_dim * 3, hidden_dim * 2),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim * 2, hidden_dim),
                                        nn.LeakyReLU(),
                                        nn.Linear(hidden_dim, z_dim * 2)
                                        )
    
    def forward(self, rgb):
        # sent rgb value to latent dimension
        z_mu, z_logvar = torch.chunk(self.sequential(rgb), 2, dim=1)
        
        return z_mu, z_logvar


class MultimodalEncoder(nn.Module):
    """
    x: text, y: color (RGB value, dim=3), z: latent
    Model p(z|x,y)
    @param embedding_module: nn.Embedding
                             pass the embedding module (share with decoder)
    @param z_dim: number of latent dimensions
    @param hidden_dim: integer [default: 256]
                       number of hidden nodes in GRU
    """
    def __init__(self, embedding_module, z_dim, rgb_dim=3, hidden_dim=256):
        super(MultimodalEncoder, self).__init__()
        assert (rgb_dim == 3)

        self.z_dim = z_dim
        self.rgb_dim = rgb_dim
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding.embedding_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.txt_lin = nn.Linear(hidden_dim, hidden_dim // 2)
        self.rgb_seq = nn.Sequential(nn.Linear(rgb_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim // 2))
        self.sequential = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                        nn.ReLU(), 
                                        nn.Linear(hidden_dim // 2, z_dim * 2))
    
    def forward(self, rgb, seq, length):
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

        txt_hidden = self.txt_lin(hidden)
        rgb_hidden = self.rgb_seq(rgb)

        concat = F.relu(torch.cat((txt_hidden, rgb_hidden), 1))
        z_mu, z_logvar = torch.chunk(self.sequential(concat), 2, dim=1)

        return z_mu, z_logvar

class ColorDecoder(nn.Module):
    """
    x: text, y: image, z: latent
    Model p(y|z)
    @param z_dim: number of latent dimensions
    """
    def __init__(self, z_dim, hidden_dim=256, rgb_dim=3):
        super(ColorDecoder, self).__init__()
        assert (rgb_dim == 3)

        self.z_dim = z_dim
        self.rgb_dim = rgb_dim
        self.hidden_dim = hidden_dim
        self.rgb_dim = rgb_dim
        self.sequential = nn.Sequential(nn.Linear(z_dim, hidden_dim), \
                                        nn.ReLU(),  \
                                        nn.Linear(hidden_dim, rgb_dim))
    
    def forward(self, z_sample):
        raw = self.sequential(z_sample)
        # print("raw and sigmoid pair: {} and {}".format(raw, torch.sigmoid(raw)))
        return torch.sigmoid(raw)