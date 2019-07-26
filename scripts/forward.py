from __future__ import print_function

import os
import sys
import random
from itertools import chain
import numpy as np
from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from color_dataset import (ColorDataset, WeakSup_ColorDataset)

from utils import (AverageMeter, save_checkpoint, _reparameterize, loss_multimodal, loss_text_unimodal, loss_image_unimodal)
from models import (TextEmbedding, TextEncoder, TextDecoder,
                    ColorEncoder, ColorEncoder_Augmented, MultimodalEncoder, ColorDecoder)

'''
	x: text, y: rgb
'''
def forward_vae_rgb_text(data_xy, models):
	y_rgb, x_tgt, x_src, x_len = data_xy
	vae_txt_enc, vae_rgb_enc, vae_mult_enc, vae_txt_dec, vae_rgb_dec = models

	batch_size = x_src.size(0)
	y_rgb = y_rgb.float()

	# Encode to |z|
	z_x_mu, z_x_logvar = vae_txt_enc(x_src, x_len)
	z_y_mu, z_y_logvar = vae_rgb_enc(y_rgb)
	z_xy_mu, z_xy_logvar = vae_mult_enc(y_rgb, x_src, x_len)

	# sample via reparametrization
	z_sample_x = _reparameterize(z_x_mu, z_x_logvar)
	z_sample_y = _reparameterize(z_y_mu, z_y_logvar)
	z_sample_xy = _reparameterize(z_xy_mu, z_xy_logvar)

	# "predictions"
	y_mu_z_y = vae_rgb_dec(z_sample_y)
	y_mu_z_xy = vae_rgb_dec(z_sample_xy)
	x_logit_z_x = vae_txt_dec(z_sample_x, x_src, x_len)
	x_logit_z_xy = vae_txt_dec(z_sample_xy, x_src, x_len)

	out = {
    		'z_x_mu': z_x_mu, 'z_x_logvar': z_x_logvar,
            'z_y_mu': z_y_mu, 'z_y_logvar': z_y_logvar,
            'z_xy_mu': z_xy_mu, 'z_xy_logvar': z_xy_logvar,
            'y_mu_z_y': y_mu_z_y, 'y_mu_z_xy': y_mu_z_xy, 
            'x_logit_z_x': x_logit_z_x, 'x_logit_z_xy': x_logit_z_xy,
            'y': y_rgb, 'x': x_tgt,'x_len': x_len
        }

	return out

def forward_vae_image_text(data_xy, models):
	'''
	x: text, y: image
	'''
	y, x_tgt, x_src, x_len = data_xy
	vae_txt_enc, vae_rgb_enc, vae_mult_enc, vae_txt_dec, vae_rgb_dec = models

	batch_size = x_src.size(0)
	y_rgb = y_rgb.float()

	# Encode to |z|
	z_x_mu, z_x_logvar = vae_txt_enc(x_src, x_len)
	z_y_mu, z_y_logvar = vae_rgb_enc(y)
	z_xy_mu, z_xy_logvar = vae_mult_enc(y, x_src, x_len)

	# sample via reparametrization
	z_sample_x = _reparameterize(z_x_mu, z_x_logvar)
	z_sample_y = _reparameterize(z_y_mu, z_y_logvar)
	z_sample_xy = _reparameterize(z_xy_mu, z_xy_logvar)

	# "predictions"
	y_mu_z_y = vae_rgb_dec(z_sample_y)
	y_mu_z_xy = vae_rgb_dec(z_sample_xy)
	x_logit_z_x = vae_txt_dec(z_sample_x, x_src, x_len)
	x_logit_z_xy = vae_txt_dec(z_sample_xy, x_src, x_len)

	out = {
		'z_x_mu': z_x_mu, 'z_x_logvar': z_x_logvar,
		'z_y_mu': z_y_mu, 'z_y_logvar': z_y_logvar,
		'z_xy_mu': z_xy_mu, 'z_xy_logvar': z_xy_logvar,
		'y_mu_z_y': y_mu_z_y, 'y_mu_z_xy': y_mu_z_xy, 
		'x_logit_z_x': x_logit_z_x, 'x_logit_z_xy': x_logit_z_xy,
		'y': y, 'x': x_tgt,'x_len': x_len
	}

	return out

def forward_vae_rgb(data_y, models):
	y_rgb = data_y[0]
	vae_rgb_enc, vae_rgb_dec = models

	batch_size = y_rgb.size(0)
	y_rgb = y_rgb.float()

	z_y_mu, z_y_logvar = vae_rgb_enc(y_rgb)
	z_sample_y = _reparameterize(z_y_mu, z_y_logvar)
	y_mu_z_y = vae_rgb_dec(z_sample_y)

	out = {
		'z_y_mu': z_y_mu, 'z_y_logvar': z_y_logvar,
		'y_mu_z_y': y_mu_z_y, 'y': y_rgb
	}

	return out

def forward_vae_image(data_y, models):
	y_tgt = data_y[0]
	vae_img_enc, vae_img_dec = models

	batch_size = y_tgt.size(0)
	y_rgb = y_rgb.float()

	z_y_mu, z_y_logvar = vae_img_enc(y_tgt)
	z_sample_y = _reparameterize(z_y_mu, z_y_logvar)
	y_mu_z_y = vae_img_dec(z_sample_y)

	out = {
		'z_y_mu': z_y_mu, 'z_y_logvar': z_y_logvar,
		'y_mu_z_y': y_mu_z_y, 'y': y_tgt
	}

	return out

def forward_vae_text(data_x, models):
	x_tgt, x_src, x_len = data_x
	vae_txt_enc, vae_txt_dec = models

	batch_size = x_src.size(0)

	z_x_mu, z_x_logvar = vae_txt_enc(x_src, x_len)
	z_sample_x = _reparameterize(z_x_mu, z_x_logvar)
	x_logit_z_x = vae_txt_dec(z_sample_x, x_src, x_len)

	out = {
		'z_x_mu': z_x_mu, 'z_x_logvar': z_x_logvar,
		'x_logit_z_x': x_logit_z_x, 'x': x_tgt, 'x_len': x_len,
	}

	return out
