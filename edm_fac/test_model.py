# Training Contrastive learning loss on timbre

import sys
import warnings
import argparse
import torch
import os
import time
import torch
import logging
warnings.simplefilter('ignore')

from modules.commons import *
from losses import *
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from dac.nn.loss import MultiScaleSTFTLoss, MelSpectrogramLoss, GANLoss, L1Loss
from audiotools import AudioSignal
from audiotools import ml
from audiotools.ml.decorators import Tracker, timer, when
from audiotools.core import util
from torch.cuda.amp import GradScaler, autocast


from dataset import EDM_Paired_Dataset
from utils import yaml_config_hook, get_infinite_loader, save_checkpoint, load_checkpoint, print_model_info
import dac

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)





def main(args):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    util.seed(args.seed)

    # Checkpoint direction
    os.makedirs(args.ckpt_path, exist_ok=True)


    generator = dac.model.MyDAC(
        encoder_dim=args.encoder_dim,
        encoder_rates=args.encoder_rates,
        latent_dim=args.latent_dim,
        decoder_dim=args.decoder_dim,
        decoder_rates=args.decoder_rates,
        sample_rate=args.sample_rate,
        timbre_classes=args.timbre_classes,
        pitch_nums=args.max_note - args.min_note + 1, # 88
    ).to(device)


    # Test DAC
    audio_data = torch.randn(2, 1, 44100).to(device)
    content_match = torch.randn(2, 1, 44100).to(device)
    timbre_match_1 = torch.randn(2, 1, 44100).to(device)
    timbre_match_2 = torch.randn(2, 1, 44100).to(device)

    output = generator.forward_contrastive(
        audio_data=audio_data,
        content_match=content_match,
        timbre_match_1=timbre_match_1,
        timbre_match_2=timbre_match_2,
    )

    print(output)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDM-FAC")

    config = yaml_config_hook("configs/config_paired.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.ckpt_path, exist_ok=True)

    main(args)
