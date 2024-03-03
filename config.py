from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function



import os.path as osp
import sys
import time
import numpy as np
import argparse
from easydict import EasyDict as edict


C = edict()
config = C
cfg = C

C.seed = 888

C.abs_dir = osp.dirname(osp.realpath(__file__))
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.repo_name = 'new_model'


C.log_dir = osp.abspath(osp.join(C.abs_dir, 'log'))
C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_dir + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.this_dir, 'models'))

"""Logging Config"""
C.wandb_activate = False

"""Dataset Config"""
C.interact_anno_dir = osp.join(C.this_dir, 'new_summary.csv')
C.motion = edict()

C.motion.interact_input_length = 25
C.motion.interact_input_length_dct = 25
C.motion.interact_target_length_train = 25
C.motion.interact_target_length_eval = 25
C.motion.dim = 54

C.use_vision = True
C.data_aug = False
C.deriv_input = True
C.deriv_output = True
C.use_relative_loss = True
C.learnable_embedding = True
C.exp_type = 'HRC'

"""Model Config"""
C.input_dim = 51*3*3
C.d_model = 512//2
C.num_layers = 2
C.nhead = 8
C.dropout = 0.2
C.activation = 'swishglu'
C.autoregressive = False

"""Train Config"""
C.batch_size = 128
C.num_workers = 2
C.lr = 0.001
C.epochs = 150
C.warmup_steps = 1000
C.training_steps = 10000
# print ("config ", C)
