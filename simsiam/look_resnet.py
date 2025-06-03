import argparse
import math
import os
import random
import shutil
import time
import warnings
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.models as models

from utils import yaml_config_hook, AverageMeter, ProgressMeter
from dataset import NewBPDataset, Transform_Pipeline
import simsiam.builder

torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    parser = argparse.ArgumentParser(description="SimSiam Single GPU")

    config = yaml_config_hook("config/ssbp_resnet50.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()


    # build model
    print("=> creating model '{}'".format(args.arch))
    model = simsiam.builder.SimSiam(
        base_encoder=models.__dict__[args.arch],
        args=args,
        dim=args.dim,
        pred_dim=args.pred_dim,
    ).cuda()
    print(model)

if __name__ == "__main__":
    main()
