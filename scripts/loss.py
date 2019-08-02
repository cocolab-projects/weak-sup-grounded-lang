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

class VAE_loss:
	def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

	def loss_multimodal(self, out, batch_size, alpha=1, beta=1, gamma=1):
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

	def loss_multimodal_only(self, out, batch_size, alpha=1, beta=1, gamma=1):
	    batch_size = out['y'].size(0)

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

	    loss = elbo_x_given_y + elbo_y_given_x
	    return loss

	def loss_text_unimodal(self, out, batch_size, alpha=1, gamma=1):
	    log_p_x_given_z_x = score_txt_logits(out['x'], out['x_logit_z_x'], out['x_len'], out['pad_index'])
	    kl_q_z_given_x_and_p_z = -0.5 * (1 + out['z_x_logvar'] - out['z_x_mu'].pow(2) - out['z_x_logvar'].exp())
	    kl_q_z_given_x_and_p_z = torch.sum(kl_q_z_given_x_and_p_z, dim=1)
	    elbo_x = alpha * -log_p_x_given_z_x + gamma * kl_q_z_given_x_and_p_z
	    elbo_x = torch.mean(elbo_x)

	    return elbo_x

	def loss_image_unimodal(self, out, batch_size, beta=1, gamma=1):
	    log_p_y_given_z_y = bernoulli_log_pdf(out['y'].view(batch_size, -1), out['y_mu_z_y'].view(batch_size, -1))
	    kl_q_z_given_y_and_p_z = -0.5 * (1 + out['z_y_logvar'] - out['z_y_mu'].pow(2) - out['z_y_logvar'].exp())
	    kl_q_z_given_y_and_p_z = torch.sum(kl_q_z_given_y_and_p_z, dim=1)
	    elbo_y = beta * -log_p_y_given_z_y + gamma * kl_q_z_given_y_and_p_z
	    elbo_y = torch.mean(elbo_y)

	    return elbo_y