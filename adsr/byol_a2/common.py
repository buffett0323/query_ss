"""BYOL for Audio: Common definitions and utilities."""

import os
import random
import numpy as np
import pandas as pd
import logging
import yaml
import datetime
import hashlib
from pathlib import Path
from easydict import EasyDict
try:
    import pickle5 as pickle
except:
    pass

import torch
from torch import nn
import torch.nn.functional as F
import torch
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as AT
import pytorch_lightning as pl


# Removed deprecated torchaudio.set_audio_backend() call since it's now a no-op with dispatcher enabled


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_timestamp():
    """ex) Outputs 202104220830"""
    return datetime.datetime.now().strftime('%y%m%d%H%M')


def load_yaml_config(path_to_config):
    """Loads yaml configuration settings as an EasyDict object."""
    path_to_config = Path(path_to_config)
    assert path_to_config.is_file()
    with open(path_to_config) as f:
        yaml_contents = yaml.safe_load(f)
    cfg = EasyDict(yaml_contents)
    return cfg


def get_logger(name):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M', level=logging.DEBUG)
    logger = logging.getLogger(name)
    return logger


def hash_text(text, L=128):
    # not used -- hashed = base64.urlsafe_b64encode(hashlib.md5(text.encode()).digest()).decode('ascii')
    hashed = hashlib.shake_128(text.encode()).hexdigest(L//2 + 1)
    return hashed[:L]
