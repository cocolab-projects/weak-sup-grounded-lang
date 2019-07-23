from __future__ import print_function
from collections import Counter, OrderedDict

import os
import sys
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal

import shutil
from tqdm import tqdm
from itertools import chain

import nltk
from nltk import sent_tokenize, word_tokenize

class AverageMeter(object):
   """Computes and stores the average and current value across groups of data

   """
   def __init__(self):
       self.reset()

   def reset(self):
       self.val = 0
       self.avg = 0
       self.sum = 0
       self.count = 0

   def update(self, val, n=1):
       self.val = val
       self.sum += val * n
       self.count += n
       self.avg = self.sum / self.count

def save_checkpoint(state, is_best, folder='./', filename='checkpoint'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename + '.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename + '.pth.tar'),
                        os.path.join(folder, filename + '_best.pth.tar'))

class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first seen'
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))
    def __reduce__(self):
        return self.__class__, (OrderedDict(self), )

def get_text(i2w, input, length):
    """Function: get_text
    Args:
        param1 (dict) i2w: from vocab. index-to-token dictionary
        param2 (PyTorch Tensor) input: sequence of indices
        param3 (int) length: length of sentence w/o padding
    Returns:
        (string): translated & concatenated tokens

    Translates a sequence (of indices) input to a string of original tokens
    """
    text = ""
    for j in range(length):
        text += " " + i2w[input[j].item()]
    return text


def hsl2rgb(hsl):
    """Function: hsl2rgb 
    Args:
        param1 (Numpy.array) hsl: contains H, S, L coordinates
    Returns:
        (tuple): tuple of translated RGB coordinate

    Convert HSL coordinates to RGB coordinates.
    https://www.rapidtables.com/convert/color/hsl-to-rgb.html
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

def _kl_normal_normal(mu1, mu2, logvar1, logvar2):
    """Function: _kl_normal_normal
    Args:
        param1 (PyTorch.Tensor) mu1: mu_p
        param2 (PyTorch.Tensor) mu2: mu_q
        param3 (PyTorch.Tensor) logvar1: logvar_p
        param3 (PyTorch.Tensor) logvar2: logvar_q
    Returns:
        (PyTorch.Tensor): KL divergence between |p| and |q|

    Computes KL divergence between two Gaussians p, q
    https://tgmstat.wordpress.com/2013/07/10/kullback-leibler-divergence/
    Equation:
        D_KL(P || Q) = SUM(P(x)(logP(x) - logQ(x))
                     = log (var_p / var_q) + (var_p^2 + (mu_p - mu_q)^2) / (2 * var_q) - 1/2
    """
    var1, var2 = torch.exp(logvar1), torch.exp(logvar2)
    return 0.5 * (((mu1 - mu2)**2 + var1 - var2)/var2 + logvar2 - logvar1)

def _reparameterize(mu, logvar):
    """Function: reparameterize
    Args:
        param1 (PyTorch.Tensor) mu: mean for latent variable |z|
        param2 (PyTorch.Tensor) logvar: logvar for latent variable |z|
    Returns:
        (PyTorch.Tensor): reparameterized sample

    Samples from standard Gaussain and reparameterizes
    Equation:
        epsilon ~ N(0,1)
        epsilon * var^(1/2) + mean
    """
    epsilon = torch.randn_like(mu)
    return torch.exp(0.5 * logvar) * epsilon + mu

def _log_mean_exp(x, dim=1):
    """log(1/k * sum(exp(x))): this normalizes x.
    @param x: PyTorch.Tensor
              samples from gaussian
    @param dim: integer (default: 1)
                which dimension to take the mean over
    @return: PyTorch.Tensor
             mean of x
    """
    m = torch.max(x, dim=dim, keepdim=True)[0]
    return m + torch.log(torch.mean(torch.exp(x - m),
                         dim=dim, keepdim=True))

def score_txt_logits(text_seq, text_logits, text_len, ignore_index):
    n, s, v = text_logits.size()
    text_logits_2d = text_logits.contiguous().view(n * s, v)
    text_seq_2d = text_seq[:, :s].contiguous().view(n * s)
    loss = -F.cross_entropy(text_logits_2d, text_seq_2d, 
                            ignore_index=ignore_index, reduction='none')
    loss = loss.view(n, s)
    loss = torch.sum(loss, dim=1)
    return loss
    # return loss / text_len.float()

def bernoulli_log_pdf(x, mu):
    mu = torch.clamp(mu, 1e-7, 1.-1e-7)
    return torch.sum(x * torch.log(mu) + (1. - x) * torch.log(1. - mu), dim=1)

def gaussian_log_pdf(x, mu, logvar):
    sigma = torch.exp(0.5 * logvar)
    dist = Normal(mu, sigma)
    return dist.log_prob(x)

def isotropic_gaussian_log_pdf(x):
    mu = torch.zeros_like(x)
    logvar = torch.zeros_like(x)
    return gaussian_log_pdf(x, mu, logvar)

def loss_multimodal(out, batch_size, alpha=1, beta=1, gamma=1):
    batch_size = out['y'].size(0)

    log_p_x_given_z_x = score_txt_logits(out['x'], out['x_logit_z_x'], out['x_len'], out['pad_index'])
    kl_q_z_given_x_and_p_z = -0.5 * (1 + out['z_x_logvar'] - out['z_x_mu'].pow(2) - out['z_x_logvar'].exp())
    kl_q_z_given_x_and_p_z = torch.sum(kl_q_z_given_x_and_p_z, dim=1)
    elbo_x = alpha * -log_p_x_given_z_x + gamma * kl_q_z_given_x_and_p_z
    elbo_x = torch.mean(elbo_x)

    log_p_y_given_z_y = bernoulli_log_pdf(out['y'].view(batch_size, -1), out['y_mu_z_y'].view(batch_size, -1))
    kl_q_z_given_y_and_p_z = -0.5 * (1 + out['z_y_logvar'] - out['z_y_mu'].pow(2) - out['z_y_logvar'].exp())
    kl_q_z_given_y_and_p_z = torch.sum(kl_q_z_given_y_and_p_z, dim=1)
    elbo_y = beta * -log_p_y_given_z_y + gamma * kl_q_z_given_y_and_p_z
    elbo_y = torch.mean(elbo_y)

    log_p_x_given_z_xy = score_txt_logits(out['x'], out['x_logit_z_xy'], out['x_len'], out['pad_index'])
    kl_q_z_given_xy_q_z_given_y = _kl_normal_normal(out['z_xy_mu'], out['z_y_mu'], out['z_xy_logvar'], out['z_y_logvar'])
    kl_q_z_given_xy_q_z_given_y = torch.sum(kl_q_z_given_xy_q_z_given_y, dim=1)
    elbo_x_given_y = alpha * -log_p_x_given_z_xy + gamma * kl_q_z_given_xy_q_z_given_y
    elbo_x_given_y = torch.mean(elbo_x_given_y)

    log_p_y_given_z_xy = bernoulli_log_pdf(out['y'].view(batch_size, -1), out['y_mu_z_xy'].view(batch_size, -1))
    kl_q_z_given_xy_q_z_given_x = _kl_normal_normal(out['z_xy_mu'], out['z_x_mu'], out['z_xy_logvar'], out['z_x_logvar'])
    kl_q_z_given_xy_q_z_given_x = torch.sum(kl_q_z_given_xy_q_z_given_x, dim=1)
    elbo_y_given_x = beta * -log_p_y_given_z_xy + gamma * kl_q_z_given_xy_q_z_given_x
    elbo_y_given_x = torch.mean(elbo_y_given_x)

    loss = elbo_x + elbo_y + elbo_x_given_y + elbo_y_given_x
    return loss

def get_image_text_joint_nll(y, y_mu_list, x_tgt, x_tgt_logits_list, x_len, z_list, z_mu, z_logvar, pad_index, verbose=False):
    batch_size = y.size(0)
    N = len(x_tgt_logits_list)
    log_p_xy_list = []

    extended_dim = [N] + [1] * y.dim()
    y, x_tgt = y.unsqueeze(0).repeat(extended_dim), x_tgt.unsqueeze(0).repeat(N, 1)
    z_mu, z_logvar = z_mu.unsqueeze(0).repeat(N, 1), z_logvar.unsqueeze(0).repeat(N, 1)

    log_p_x_given_z = score_txt_logits(x_tgt, x_tgt_logits_list, x_len, pad_index)
    log_p_y_given_z = bernoulli_log_pdf(y.float().view(N, -1), y_mu_list.view(N, -1))

    ## *** should be torch.sum since this is log of products, but it is summed over |z_dim| and thus magnitude is very large ***
    ## this trivializes the role of p(x|z) and p(y|z), so to adjust I use torch.mean, effectively dividing by 100
    log_q_z_given_xy = torch.sum(gaussian_log_pdf(z_list, z_mu, z_logvar), dim=1)
    log_p_z = torch.sum(isotropic_gaussian_log_pdf(z_list), dim=1)
    # log_q_z_given_xy = torch.mean(gaussian_log_pdf(z_list, z_mu, z_logvar), dim=1)
    # log_p_z = torch.mean(isotropic_gaussian_log_pdf(z_list), dim=1)   

    log_p_xy = log_p_x_given_z + log_p_y_given_z + log_p_z - log_q_z_given_xy
    log_p_xy = log_p_xy.cpu()  # cast to CPU so we don't blow up

    nll = _log_mean_exp(log_p_xy.unsqueeze(0), dim=1)
    nll = -torch.mean(nll)

    if verbose and N == 1:
        print("x_len: {}".format(x_len))
        print("log_p_x_given_z: {} for x_tgt: {}".format(log_p_x_given_z, x_tgt))
        print("log_p_y_given_z: {}".format(log_p_y_given_z))
        print("log_p_z: {}".format(log_p_z))
        print("log_q_z_given_xy: {}".format(log_q_z_given_xy))
        print()
        # breakpoint()

    return nll

def preprocess_text(text):
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

