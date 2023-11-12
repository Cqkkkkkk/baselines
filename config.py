import logging
import os
from yacs.config import CfgNode as CN


# Global config object
cfg = CN()

def set_cfg(cfg):
    r'''
     This function sets the default config value.
     1) Note that for an experiment, only part of the arguments will be used
     The remaining unused arguments won't affect anything.
     2) We support *at most* two levels of configs, e.g., cfg.dataset.name
     '''

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    
    # Select the device, cpu or cuda 
    cfg.device = 'cuda:1'
    
    # Random seed
    cfg.seed = 42

    # Repeat experitment times
    cfg.repeat = 1

    # ------------------------------------------------------------------------ #
    # Dataset options
    # ------------------------------------------------------------------------ #

    cfg.dataset = CN()

    cfg.dataset.name = 'cora'

    # Modified automatically by code, no need to set
    cfg.dataset.num_nodes = -1

    # Modified automatically by code, no need to set
    cfg.dataset.num_classes = -1

    # Dir to load the dataset. If the dataset is downloaded, it is in root
    cfg.dataset.root = './datasets'

    # Assert split in ['public', 'random']
    cfg.dataset.split = 'public'
    
    # Only works if split='ramdom' is set, train-val-(test)
    cfg.dataset.random_split = [0.6, 0.2] 


    # ------------------------------------------------------------------------ #
    # Optimization options
    # ------------------------------------------------------------------------ #

    cfg.optim = CN()

    # Maximal number of epochs
    cfg.optim.epochs = 50

    cfg.optim.patience = 30
   

    # Base learning rate
    cfg.optim.lr = 0.01

    # L2 regularization
    cfg.optim.wd = 5e-4

    # Batch size, only works in minibatch mode
    cfg.optim.batch_size = 10000

    # Sampled neighbors size, only works in minibatch mode
    cfg.optim.num_neighbors = [15, 2]


    # ------------------------------------------------------------------------ #
    # Model options 
    # ------------------------------------------------------------------------ #
    
    cfg.model = CN()

     # Model to use 
    cfg.model.name = 'gcn'
    
    # Hidden layer dim 
    cfg.model.hidden_dim = 64

    # Number of attetnion heads
    cfg.model.num_heads = 8

    # Layer number
    cfg.model.num_layers = 2

    # Dropout rate
    cfg.model.dropout = 0.5


set_cfg(cfg)