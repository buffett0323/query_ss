import re
import math
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio.transforms as T
import torchvision.transforms as transforms
import nnAudio.features

from pathlib import Path
from swin_transformer import SwinTransformer
from utils import *
from audio_ntt import AudioNTT2020Task6X
from torch_models import Cnn14_16k
from conv_next import ConvNeXt
from dataset import SegmentBPDataset
from augmentation import PrecomputedNorm, NormalizeBatch



class SimSiam(nn.Module):
    """
    SimSiam model with Wavegram_Logmel128_Cnn14 as the encoder.
    """
    def __init__(
        self, 
        args, 
        dim=2048, 
        pred_dim=512,
    ):
        super(SimSiam, self).__init__()
        self.args = args

        # ** Encoders **
        if self.args.encoder_name == "AudioNTT2022Encoder":
            self.encoder = AudioNTT2020Task6X(
                n_mels=self.args.n_mels,
                d=dim,
            )
            prev_dim = self.encoder.d

            self.encoder.fc2 = nn.Sequential(
                nn.Dropout(p=0.3, inplace=False),
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(inplace=True), # first layer
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(inplace=True), # second layer
                nn.Linear(prev_dim, prev_dim, bias=True),
                nn.BatchNorm1d(dim, affine=False)
            ) # output layer
            self.encoder.fc2[7].bias.requires_grad = False # hack: not use bias as it is followed by BN

            
        elif self.args.encoder_name == "SwinTransformer":
            self.encoder = SwinTransformer(
                img_size=args.img_size, 
                window_size=args.swint_window_size, 
                in_chans=args.channels, 
                num_classes=dim,  # num_classes = 0 --> self.head = nn.Identity
            )
            prev_dim = self.encoder.num_features
            
            # **Build a separate 3-layer projector**
            self.encoder.head = nn.Sequential(
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(inplace=True), # first layer
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(inplace=True), # second layer
                self.encoder.head,
                nn.BatchNorm1d(dim, affine=False)
            ) # output layer
            self.encoder.head[6].bias.requires_grad = False # hack: not use bias as it is followed by BN
            
        
        elif self.args.encoder_name == "ConvNeXt":
            if args.convnext_model == "tiny":
                depths = [3, 3, 9, 3]
                dims = [96, 192, 384, 768]
            elif args.convnext_model == "small":
                depths = [3, 3, 27, 3]
                dims = [96, 192, 384, 768]
            elif args.convnext_model == "base":
                depths = [3, 3, 27, 3]
                dims = [128, 256, 512, 1024]
            elif args.convnext_model == "large":
                depths = [3, 3, 27, 3]
                dims = [192, 384, 768, 1536]
            else:
                raise ValueError(f"Invalid model: {args.convnext_model}")
            
            print("Using ConvNeXt model: {}".format(args.convnext_model))
                
            self.encoder = ConvNeXt(
                in_chans=args.channels,
                num_classes=dim,
                depths=depths,
                dims=dims,
            )
            prev_dim = self.encoder.head.weight.shape[1]
            print("prev_dim", prev_dim)
            
            # **Build a separate 3-layer projector**
            self.encoder.head = nn.Sequential(
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(inplace=True), # first layer
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(inplace=True), # second layer
                self.encoder.head,
                nn.BatchNorm1d(dim, affine=False)
            ) # output layer
            self.encoder.head[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

            
        else:
            self.encoder = nn.Identity()
        
        
        # **Build a 2-layer predictor**
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # Hidden layer
            nn.Linear(pred_dim, dim)  # Output layer
        )
        

    def forward(self, x1, x2):
        # Compute features for both views
        z1 = self.encoder(x1) # self.projector(self.encoder(x1))  # NxC
        z2 = self.encoder(x2) # self.projector(self.encoder(x2))  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1.detach(), z2.detach()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR Encoder")

    config = yaml_config_hook("config/ssbp_convnext.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    
    # Load models
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = SimSiam(
        args=args,
        dim=args.dim,
        pred_dim=args.pred_dim,
    ).to(device)
    
    
    train_dataset = SegmentBPDataset(
        data_dir=args.seg_dir,
        split="train",
        stem="other",
        eval_mode=False,
        train_mode=args.train_mode,
        p_ts=args.p_ts,
        p_ps=args.p_ps,
        p_tm=args.p_tm,
        p_tstr=args.p_tstr,
        semitone_range=args.semitone_range,
        tm_min_band_part=args.tm_min_band_part,
        tm_max_band_part=args.tm_max_band_part,
        tm_fade=args.tm_fade,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16, #args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
        persistent_workers=args.persistent_workers,
        prefetch_factor=8, #4,
    )
    
    to_spec = nnAudio.features.MelSpectrogram(
        sr=args.sample_rate,
        n_fft=args.n_fft,
        win_length=args.window_size,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        center=True,
        power=2,
        verbose=False,
    ).to(device)
    
    pre_norm = PrecomputedNorm(np.array(args.norm_stats)).to(device)
    post_norm = NormalizeBatch().to(device)

    for x_i, x_j, _, _ in train_loader:
        x_i = x_i.to(device, non_blocking=True)
        x_j = x_j.to(device, non_blocking=True)
        
        x_i = (to_spec(x_i) + torch.finfo().eps).log()
        x_i = pre_norm(x_i).unsqueeze(1)
        x_j = (to_spec(x_j) + torch.finfo().eps).log()
        x_j = pre_norm(x_j).unsqueeze(1)
        
        bs = x_i.shape[0]
        paired_inputs = torch.cat([x_i, x_j], dim=0)
        paired_inputs = post_norm(paired_inputs)
        
        p1, p2, z1, z2 = model(x1=paired_inputs[:bs], x2=paired_inputs[bs:])
        print(z1, z2)
        break