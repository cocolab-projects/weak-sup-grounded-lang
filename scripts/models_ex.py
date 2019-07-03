from __future__ import print_function

import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.nn.utils.rnn as rnn_utils
from dc_models import gan_block, gen_32_conv_output_dim


class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256):
        super(TextEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
    
    def forward(self, x):
        return self.embedding(x)


class TextEncoder(nn.Module):
    r"""Parameterizes q(z|sentence) where we use a recurrent
    model to project a sentence into a vector space.
    @param embedding_module: nn.Embedding
                             pass the embedding module (share with
                             decoder)
    @param z_dim: integer
                  number of latent dimensions
    @param hidden_dim: integer [default: 256]
                       number of hidden nodes in GRU
    """
    def __init__(self, embedding_module, z_dim, hidden_dim=256):
        super(TextEncoder, self).__init__()
        self.z_dim = z_dim
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding.embedding_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, self.z_dim * 2)

    def forward(self, seq, length):
        batch_size = seq.size(0)

        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]

        # embed your sequences
        embed_seq = self.embedding(seq)

        # reorder from (B,L,D) to (L,B,D)
        embed_seq = embed_seq.transpose(0, 1)

        packed = rnn_utils.pack_padded_sequence(
            embed_seq,
            sorted_lengths.data.tolist() if batch_size > 1 else length.data.tolist())

        _, hidden = self.gru(packed)
        hidden = hidden[-1, ...]

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]

        z_mu, z_logvar = torch.chunk(self.linear(hidden), 2, dim=1)

        return z_mu, z_logvar


class ImageEncoder(nn.Module):


class TextDecoder(nn.Module):
    """Parameterizes p(sentence|z) where we use a recurrent
    model to generate a distribution of a sequence of tokens.
    Assumes a maximum sequence length and a fixed vocabulary.
    We return logits to a categorical so please use
        nn.CrossEntropy
    instead of
        nn.NLLLoss
    @param embedding_module: nn.Embedding
                             pass the embedding module (share with
                             decoder)
    @param z_dim: integer
                  number of latent dimensions
    @param hidden_dim: integer [default: 256]
                       number of hidden nodes in GRU
    @param word_dropout: float [default: 0]
                         with some probability, drop tokens
                         so we force usage of z
    @param embedding_dropout: float [default: 0]
                              with  some probability delete entries
                              of the input embedding
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
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim)
        self.outputs2vocab = nn.Linear(self.hidden_dim, self.vocab_size)
        self.latent2hidden = nn.Linear(self.z_dim, self.hidden_dim)
        self.word_dropout = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

    def forward(self, z, seq, length):
        batch_size = z.size(0)

        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)

            z = z[sorted_idx]
            seq = seq[sorted_idx]
        else:
            sorted_lengths = length  # since we use this variable later

        if self.word_dropout > 0:
            # randomly replace with unknown tokens
            prob = torch.rand(seq.size())
            prob[(seq.cpu().data - self.sos_index) & \
                 (seq.cpu().data - self.pad_index) == 0] = 1
            mask_seq = seq.clone()
            mask_seq[(prob < self.word_dropout).to(z.device)] = self.unk_index
            seq = mask_seq

        # embed your sequences
        embed_seq = self.embedding(seq)
        # paper says this dropout doesn't really help.
        embed_seq = self.embedding_dropout(embed_seq)

        # reorder from (B,L,D) to (L,B,D)
        embed_seq = embed_seq.transpose(0, 1)

        packed_input = rnn_utils.pack_padded_sequence(embed_seq, sorted_lengths)

        # initialize hidden state
        hidden = self.latent2hidden(z)
        hidden = hidden.unsqueeze(0).contiguous()

        # shape = (seq_len, batch, hidden_dim)
        packed_output, _ = self.gru(packed_input, hidden)
        output = rnn_utils.pad_packed_sequence(packed_output)
        output = output[0].contiguous()

        # reorder from (L,B,D) to (B,L,D)
        output = output.transpose(0, 1)

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            output = output[reversed_idx]

        max_length = output.size(1)
        output_2d = output.view(batch_size * max_length, self.hidden_dim)
        outputs_2d = self.outputs2vocab(output_2d)
        outputs = outputs_2d.view(batch_size, max_length, self.vocab_size)

        return outputs

    def sample(self, z, max_seq_length, greedy=False):
        """Sample tokens in an auto-regressive framework."""
        with torch.no_grad():
            batch_size = z.size(0)

            # initialize hidden state
            hidden = self.latent2hidden(z)
            hidden = hidden.unsqueeze(0).contiguous()

            # first input is SOS token
            inputs = np.array([self.sos_index for _ in xrange(batch_size)])
            inputs = torch.from_numpy(inputs)
            inputs = inputs.unsqueeze(1)
            inputs = inputs.to(z.device)

            # save SOS as first generated token
            inputs_npy = inputs.squeeze(1).cpu().numpy()
            sampled_ids = [[w] for w in inputs_npy]

            # (B,L,D) to (L,B,D)
            inputs = inputs.transpose(0, 1)

            # compute embeddings
            inputs = self.embedding(inputs)

            for i in xrange(max_seq_length):
                outputs, hidden = self.gru(inputs, hidden)  # outputs: (L=1,B,H)
                outputs = outputs.squeeze(0)                # outputs: (B,H)
                outputs = self.outputs2vocab(outputs)       # outputs: (B,V)

                if greedy:
                    predicted = outputs.max(1)[1]
                    predicted = predicted.unsqueeze(1)
                else:
                    outputs = F.softmax(outputs, dim=1)
                    predicted = torch.multinomial(outputs, 1)

                predicted_npy = predicted.squeeze(1).cpu().numpy()
                predicted_lst = predicted_npy.tolist()

                for w, so_far in zip(predicted_lst, sampled_ids):
                    if so_far[-1] != self.eos_index:
                        so_far.append(w)

                inputs = predicted.transpose(0, 1)          # inputs: (L=1,B)
                inputs = self.embedding(inputs)             # inputs: (L=1,B,E)

            sampled_lengths = [len(text) for text in sampled_ids]
            sampled_lengths = np.array(sampled_lengths)

            max_length = max(sampled_lengths)
            padded_ids = np.ones((batch_size, max_length)) * self.pad_index

            for i in xrange(batch_size):
                padded_ids[i, :sampled_lengths[i]] = sampled_ids[i]

            sampled_lengths = torch.from_numpy(sampled_lengths).long()
            sampled_ids = torch.from_numpy(padded_ids).long()

        return sampled_ids, sampled_lengths

    def beam_search(self, z, max_seq_length, beam_width=10):
        r"""Sample tokens but keep around a beam."""
        beam_inputs = []
        beam_hiddens = []
        beam_scores = []
        beam_sampled_ids = []

        with torch.no_grad():
            # ----- this section is deterministic -----
            batch_size = z.size(0)

            # initialize hidden state
            hidden = self.latent2hidden(z)
            hidden = hidden.unsqueeze(0).contiguous()

            # first input is SOS token
            inputs = np.array([self.sos_index for _ in xrange(batch_size)])
            inputs = torch.from_numpy(inputs)
            inputs = inputs.unsqueeze(1)
            inputs = inputs.to(z.device)

            # save SOS as first generated token
            inputs_npy = inputs.squeeze(1).cpu().numpy()
            sampled_ids = [[w] for w in inputs_npy]

            # (B,L,D) to (L,B,D)
            inputs = inputs.transpose(0, 1)

            # compute embeddings
            inputs = self.embedding(inputs)

            for _ in xrange(beam_width):
                beam_scores.append(0.)  # blank slate at this point
                beam_sampled_ids.append(deepcopy(sampled_ids))

            # we do one step so we can distinguish these different values
            outputs, hidden = self.gru(inputs, hidden)
            outputs = outputs.squeeze(0)
            outputs = self.outputs2vocab(outputs)
            log_dist = F.log_softmax(outputs, dim=1)
            log_dist = log_dist.squeeze(0)
            outputs_npy = outputs.squeeze(0).cpu().detach().numpy()
            order = np.argsort(outputs_npy)[::-1]

            for k in xrange(beam_width):
                predicted_lst = [order[k]]
                predicted = torch.Tensor([order[k]]).unsqueeze(1)
                predicted = predicted.to(z.device)
                predicted = predicted.long()
                predicted_logproba = log_dist[order[k]]

                for w, so_far in zip(
                    predicted_lst, beam_sampled_ids[k]
                ):
                    if so_far[-1] != self.eos_index:
                        so_far.append(w)

                inputs = predicted.transpose(0, 1)
                inputs = self.embedding(inputs)

                beam_inputs.append(inputs.clone())
                beam_hiddens.append(hidden.clone())
                beam_scores[k] += float(predicted_logproba)

            # ----- end deterministic section -----

            for i in xrange(max_seq_length - 1):
                _beam_inputs = []
                _beam_hiddens = []
                _beam_scores = [deepcopy(beam_scores[j]) for _ in xrange(beam_width) 
                                for j in xrange(beam_width)]
                _beam_scores = np.array(_beam_scores)
                _beam_sampled_ids = [deepcopy(beam_sampled_ids[j]) for j in xrange(beam_width)
                                     for _ in xrange(beam_width)]

                for j in xrange(beam_width):
                    inputs = beam_inputs[j]
                    hidden = beam_hiddens[j]

                    outputs, hidden = self.gru(inputs, hidden)  # outputs: (L=1,B,H)
                    outputs = outputs.squeeze(0)                # outputs: (B,H)
                    outputs = self.outputs2vocab(outputs)       # outputs: (B,V)
                    log_dist = F.log_softmax(outputs, dim=1)    # outputs: (B, V)
                                                                # distribution over elements
                    log_dist = log_dist.squeeze(0)
                    outputs_npy = outputs.squeeze(0).cpu().detach().numpy()

                    # define ordering by local logits
                    order = np.argsort(outputs_npy)[::-1]

                    for k in xrange(beam_width):
                        predicted_lst = [order[k]]
                        predicted = torch.Tensor([order[k]]).unsqueeze(1)
                        predicted = predicted.to(z.device)
                        predicted = predicted.long()
                        predicted_logproba = log_dist[order[k]]

                        for w, so_far in zip(
                            predicted_lst, _beam_sampled_ids[j * beam_width + k]
                        ):
                            if so_far[-1] != self.eos_index:
                                so_far.append(w)

                        inputs = predicted.transpose(0, 1)     # inputs: (L=1,B)
                        inputs = self.embedding(inputs)        # inputs: (L=1,B,E)

                        _beam_inputs.append(inputs.clone())
                        _beam_hiddens.append(hidden.clone())
                        _beam_scores[j * beam_width + k] += float(predicted_logproba)

                # define ordering by global probabilities
                order = np.argsort(_beam_scores)[::-1]

                for j in xrange(beam_width):
                    beam_inputs[j] = _beam_inputs[order[j]]
                    beam_hiddens[j] = _beam_hiddens[order[j]]
                    beam_scores[j] = _beam_scores[order[j]]
                    beam_sampled_ids[j] = _beam_sampled_ids[order[j]]

            # ----- end beam search -----

            final_sampled_ids = [] 
            final_sampled_lengths = []
            final_sampled_logprobas = []

            for j in xrange(beam_width):
                sampled_ids = beam_sampled_ids[j]
                sampled_logprobas = beam_scores[j]
                sampled_lengths = [len(text) for text in sampled_ids]
                sampled_lengths = np.array(sampled_lengths)

                max_length = max(sampled_lengths)
                padded_ids = np.ones((batch_size, max_length)) * self.pad_index

                for i in xrange(batch_size):
                    padded_ids[i, :sampled_lengths[i]] = sampled_ids[i]

                sampled_lengths = torch.from_numpy(sampled_lengths).long()
                sampled_ids = torch.from_numpy(padded_ids).long()

                final_sampled_ids.append(sampled_ids)
                final_sampled_lengths.append(sampled_lengths)
                final_sampled_logprobas.append(sampled_logprobas)

        return final_sampled_ids, final_sampled_lengths, final_sampled_logprobas
