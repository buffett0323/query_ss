import torch
import torch.nn as nn
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from utils import *
from torch_models import ConvBlock
import argparse
import torchlibrosa as tl

# XXX torch.Size([16, 132300])
# XXA1A1: torch.Size([16, 64, 205, 64]) torch.Size([16, 64, 206, 64])


parser = argparse.ArgumentParser(description="SimCLR_BP")

config = yaml_config_hook("bp_config.yaml")
for k, v in config.items():
    parser.add_argument(f"--{k}", default=v, type=type(v))

args = parser.parse_args()


batch_size = 16
sample_rate = 44100 #22050
win_length = 2048
hop_length = 512
n_mels = 128

x = torch.empty(batch_size, sample_rate*3).uniform_(-1, 1)  # (batch_size, sample_rate)
print(x.shape)


# TorchLibrosa feature extractor the same as librosa.feature.melspectrogram()
feature_extractor = torch.nn.Sequential(
    tl.Spectrogram(
        hop_length=hop_length,
        win_length=win_length,
    ),
    tl.LogmelFilterBank(
        sr=sample_rate,
        n_mels=n_mels,
        is_log=False, # Default is true
    ))
batch_feature = feature_extractor(x) # (batch_size, 1, time_steps, mel_bins)
print(batch_feature.shape)


spectrogram_extractor = Spectrogram(
    n_fft=args.n_fft,
    hop_length=args.hop_length,
    win_length=args.window_size,
    window='hann', #args.window,
    center=True, #args.center,
    pad_mode='reflect', #args.pad_mode,
    freeze_parameters=True
)

logmel_extractor = LogmelFilterBank(
    sr=args.sample_rate, n_fft=args.n_fft,
    n_mels=args.n_mels, fmin=args.fmin, fmax=args.fmax,
    ref=1.0, amin=1e-10, top_db=None,
    freeze_parameters=True
)

bn0 = nn.BatchNorm2d(128)
spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=16, freq_stripes_num=2)

conv_block1 = ConvBlock(in_channels=1, out_channels=64)


x = spectrogram_extractor(x)
x = logmel_extractor(x)
x = x.transpose(1, 3)
x = bn0(x)
x = x.transpose(1, 3)
x = spec_augmenter(x)
x = conv_block1(x, pool_size=(2, 2), pool_type='avg')
print(x.shape)
