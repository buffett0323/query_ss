import math
import librosa
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from models.e2e.bandit.bandsplit import BandSplitModule
from models.e2e.bandit.utils import MusicalBandsplitSpecification
from models.e2e.bandit.tfmodel import SeqBandModellingModule
from models.e2e.querier.passt import Passt, PasstWrapper
from models.e2e.base import BaseEndToEndModule
from models.types import InputType, OperationMode, SimpleishNamespace
from beats.BEATs import BEATs, BEATsConfig

from unet import UnetIquery
from film import FiLM


if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)

    

""" T = [(x - win_length) / hop_length] """
SET_LENGTH = 261888


class MyModel(nn.Module):
    def __init__(
        self,
        batch_size,
        in_channels, 
        embedding_size, 
        out_channels, 
        fs, 
        kernel_size, 
        stride,
        n_masks=2,
        n_fft=1022,
        hop_length=256,
        win_length=1022,
        n_sqm_modules=12,
        rnn_dim=256,
        eps=1e-10,
        bidirectional=True,
        rnn_type="LSTM",
        mix_query_mode="FiLM",
    ):
        super(MyModel, self).__init__()
        
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.embedding_size = embedding_size
        self.out_channels = out_channels
        self.fs = fs
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_masks = n_masks
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.mix_query_mode = mix_query_mode
        self.F = 512
        self.T = 256
        self.eps = eps
        
        
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad_mode="constant",
            pad=0,
            window_fn=torch.__dict__["hann_window"],
            wkwargs=None,
            power=None,
            normalized=True,
            center=True,
            onesided=True,
        )
        
        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad_mode="constant",
            pad=0,
            window_fn=torch.__dict__["hann_window"],
            wkwargs=None,
            normalized=True,
            center=True,
            onesided=True,
        )

        self.film = FiLM(
            cond_embedding_dim=256, 
            channels=256, 
            additive=True, 
            multiplicative=True
        )
        
        
        self.instantiate_beats(beats_check_point_pth='beats/pt_dict/BEATs_iter3_plus_AS2M.pt')
        
        self.query_encoder = QueryEncoder(
            in_channels=embedding_size,
            hidden_channels=512,
            out_channels=256,
        ) # 768 -> 256
        
        self.unet = UnetIquery(
            fc_dim=64, 
            num_downs=5, 
            ngf=64, 
            use_dropout=False,
        )
        
        self.mlp = MLP(
            input_dim=self.T * 128, 
            hidden_dim=512, 
            output_dim=self.n_masks, 
            num_layers=3,
        )


    def instantiate_beats(
        self,
        beats_check_point_pth='beats/pt_dict/BEATs_iter3_plus_AS2M.pt',
    ):
        checkpoint = torch.load(beats_check_point_pth)

        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint['model'])
        self.beats = BEATs_model.eval().cuda()

        
    def adapt_query(self, wav):
        padding_mask = torch.zeros(1, wav.shape[1]).bool().cuda()  # Move padding mask to GPU
        embed = []
        
        for i in range(self.batch_size):
            embed.append(self.beats.extract_features(wav[i].unsqueeze(0), padding_mask=padding_mask)[0].mean(dim=1, keepdim=False))

        embed = torch.cat(embed, dim=0)
        embed = self.query_encoder(embed)
        return embed
    
    def mask(self, x, m):
        return x * m
    
    def pre_process(self, batch: InputType):
        """
            Transform from Binaural into Mono
            Both the mixture and the stems
        """
        batch.mixture.audio = batch.mixture.audio[:, :, :SET_LENGTH].mean(dim=1, keepdim=True)

        # Compute the STFT spectrogram
        with torch.no_grad():
            batch.mixture.spectrogram = self.stft(batch.mixture.audio)
            
            if "sources" in batch.keys():
                for stem in batch.sources.keys():
                    batch.sources[stem].audio = batch.sources[stem].audio[:, :, :SET_LENGTH].mean(dim=1, keepdim=True)
                    batch.sources[stem].spectrogram = self.stft(batch.sources[stem].audio)
            
            batch.query.audio = batch.query.audio.mean(dim=1, keepdim=False)
                    
        return batch
    
    
    
    def forward(self, batch: InputType): # def forward(self, x, Z, tgt):
        
        """
        input shape:
            x: [Batch size, C, N]
            Z: [Batch size, D=768] (Query)
            tgt: target stem audio
        """
        
        # Preprocessing
        batch = self.pre_process(batch)

        # Separate mixture spec -> unet
        x = batch.mixture.spectrogram
        x = torch.abs(x)
        x, x_latent = self.unet(x)

        # Query encoder
        Z = self.adapt_query(batch.query.audio)

        
        """
            Ways to Combine Mixture & Query
            1. FiLM Condition + MLP
            2. Transformer Self attention or Cross attention
        """
        
        # First Way: FiLM Condition + MLP
        x_latent = self.film(x_latent, Z)
        x_latent = x_latent.permute(0, 2, 1, 3) # torch.Size([BS, 64, 256, 32*4])
        x_latent = self.mlp(x_latent) # ([BS=2, C_e=64, N=2])
        
        # Second Way: Transformer Self Attention
        """ SKIP First """
        
        # Mask estim
        mask = torch.einsum('bcft,bcn->bnft', x, x_latent) # torch.Size([4, 2, 512, 256])
        s = self.mask(batch.mixture.spectrogram, mask)
        
        # Write the predicted results back to batch data
        batch.estimates["target"] = SimpleishNamespace(
            audio=self.istft(s[:,0,:,:]) #, spectrogram=s[:,0,:,:]
        )
        batch.estimates["non-target"] = SimpleishNamespace(
            audio=self.istft(s[:,1,:,:]) #, spectrogram=s[:,1,:,:]
        )
        
        # # Calculate the mask for the other stems and get the ground truth spectrogram
        # mask = torch.abs(tgt) / (torch.abs(x) + 1e-10)
        # gt_stft = torch.stack((tgt, (x - (mask * x))), dim=1)
        # S_mix = x # (BS=2, F=512, T=256)
        
        # # Get mixture through the U-NET
        # x = torch.abs(x).unsqueeze(1)
        # x, x_latent = self.unet(x) # ([BS=2, C_e=64, F=512, T=256]), torch.Size([BS, 256, 64, 32])
        
        # # Query Encoder
        # Z = self.query_encoder(Z) # 768 -> 256 # torch.Size([BS, 256])
        
        
        # x = self.stft(x)
        # tgt = self.stft(tgt)
        
        # # Mask dot products
        # pred_masks = torch.einsum('bcft,bcn->bnft', x, x_latent) # torch.Size([4, 2, 512, 256])
        # gt_masks = gt_stft / (S_mix.unsqueeze(1) + self.eps)

        # # Mix the mask and original mixture
        # S_k = S_mix.unsqueeze(1) * pred_masks
        # S_k = self.istft(S_k)
        # gt_stft = self.istft(gt_stft)
        
        
        # Predicted_masks, ground_truth masks, ground_truth audio, source_k (predicted)
        # return pred_masks, gt_masks, gt_stft, S_k
        return batch





class QueryEncoder(nn.Module):
    def __init__(
        self,
        in_channels=768,
        hidden_channels=512,
        out_channels=256,
        ):
        super(QueryEncoder, self).__init__()
        
        # First fully connected layer to reduce the dimension
        self.fc1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1),
            nn.ReLU()
        )
        
        # Second fully connected layer to further reduce the dimension
        self.fc2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        # Input shape is (batch_size, 768) --> (batch_size, 768, 1)
        x = x.unsqueeze(-1)
        
        x = self.fc1(x)  # Shape will become (batch_size, 512, 1)
        x = self.fc2(x)  # Shape will become (batch_size, 256, 1)
        x = x.squeeze(-1)  # Final shape will be (batch_size, 256)
        
        return x



class MLP(nn.Module):
    """Multi-layer perceptron with support for spatial dimensions."""

    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        output_dim, 
        num_layers=3,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            [nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])]
        )

    def forward(self, x):
        batch_size, C_e, T, W = x.shape  # [Batch_size, C_e, T, W]
        
        # Use reshape instead of view to flatten the spatial dimensions (T and W)
        x = x.reshape(batch_size, C_e, -1)  # New shape: [Batch_size, C_e, T * W]
        
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        
        return x
    
    
    

class UnetIquery(nn.Module):
    def __init__(self, fc_dim=64, num_downs=7, ngf=64, use_dropout=False):
        super(UnetIquery, self).__init__()
        self.bn0 = nn.BatchNorm2d(1)
        self.downrelu2 = nn.LeakyReLU(0.2, True)
        self.downrelu3 = nn.LeakyReLU(0.2, True)
        self.downrelu4 = nn.LeakyReLU(0.2, True)
        self.downrelu5 = nn.LeakyReLU(0.2, True)
        self.downrelu6 = nn.LeakyReLU(0.2, True)
        self.downrelu7 = nn.LeakyReLU(0.2, True)

        self.uprelu7 = nn.ReLU(True)
        self.upsample7 = nn.Upsample(size=(16, 9), mode='bilinear', align_corners=True) # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.uprelu6 = nn.ReLU(True)
        self.upsample6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.uprelu5 = nn.ReLU(True)
        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.uprelu4 = nn.ReLU(True)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.uprelu3 = nn.ReLU(True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.uprelu2 = nn.ReLU(True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.uprelu1 = nn.ReLU(True)
        self.upsample1 = nn.Upsample(size=(1025, 576), mode='bilinear', align_corners=True) # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.use_bias = False

        self.downconv1 = nn.Conv2d(1, ngf, kernel_size=4, stride=2, padding=1, bias=self.use_bias)
        self.downconv2 = nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1, bias=self.use_bias)
        self.downnorm2 = nn.BatchNorm2d(ngf*2)
        self.downconv3 = nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1, bias=self.use_bias)
        self.downnorm3 = nn.BatchNorm2d(ngf*4)
        self.downconv4 = nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1, bias=self.use_bias)
        self.downnorm4 = nn.BatchNorm2d(ngf*8)
        self.downconv5 = nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, bias=self.use_bias)
        self.downnorm5 = nn.BatchNorm2d(ngf*8)
        self.downconv6 = nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, bias=self.use_bias)
        self.downnorm6 = nn.BatchNorm2d(ngf*8)
        self.downconv7 = nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, bias=self.use_bias)
        
        self.upconv7 = nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        self.upnorm7 = nn.BatchNorm2d(ngf*8)
        self.upconv6 = nn.Conv2d(ngf*16, ngf*8, kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        self.upnorm6 = nn.BatchNorm2d(ngf*8)
        self.upconv5 = nn.Conv2d(ngf*16, ngf*8, kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        self.upnorm5 = nn.BatchNorm2d(ngf*8)
        self.upconv4 = nn.Conv2d(ngf*16, ngf*4, kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        self.upnorm4 = nn.BatchNorm2d(ngf*4)
        self.upconv3 = nn.Conv2d(ngf*8, ngf*2, kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        self.upnorm3 = nn.BatchNorm2d(ngf*2)
        self.upconv2 = nn.Conv2d(ngf*4, ngf, kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        self.upnorm2 = nn.BatchNorm2d(ngf)
        self.upconv1 = nn.Conv2d(ngf*2, fc_dim, kernel_size=3, stride=1, padding=1, bias=self.use_bias)

    def forward(self, x):
        x = self.bn0(x)
        #layer 1 down
        #outer_nc, inner_input_nc
        x1 = self.downconv1(x)
        #layer2 down
        
        x2 = self.downrelu2(x1)
        x2 = self.downconv2(x2)
        x2 = self.downnorm2(x2)
        
        #layer3 down
       
        x3 = self.downrelu3(x2)
        x3 = self.downconv3(x3)
        x3 = self.downnorm3(x3)
       
        #layer4 down
        x4 = self.downrelu4(x3)
        x4 = self.downconv4(x4)
        x4 = self.downnorm4(x4)
        
        #layer5 down:
        x5 = self.downrelu5(x4)
        x5 = self.downconv5(x5)
        x5 = self.downnorm5(x5)
        
        #layer6 down:
        x6 = self.downrelu6(x5)
        x6= self.downconv6(x6)
        x6 = self.downnorm6(x6)
        
        #layer7 down:
        x = self.downrelu7(x6)
        x = self.downconv7(x)
        
        
        #layer7 up:
        x = self.uprelu7(x)
        x = self.upsample7(x)
        x = self.upconv7(x)
        x = self.upnorm7(x)
        

        #layer 6 up:
        x = self.uprelu6(torch.cat([x6, x], 1))
        x = self.upsample6(x)
        x = self.upconv6(x)
        x = self.upnorm6(x)
        

        #layer 5 up:
        x = self.uprelu5(torch.cat([x5, x], 1))
        x = self.upsample5(x)
        x = self.upconv5(x)
        x = self.upnorm5(x)
        x_latent = x # revised place
        
        
        #layer 4 up:
        x = self.uprelu4(torch.cat([x4, x], 1))
        x = self.upsample4(x)
        x = self.upconv4(x)
        x = self.upnorm4(x)
        # x_latent = x # original


        #layer3 up:
        x = self.uprelu3(torch.cat([x3, x], 1))
        x = self.upsample3(x)
        x = self.upconv3(x)
        x = self.upnorm3(x)

        #layer2 up:
        x = self.uprelu2(torch.cat([x2, x], 1))
        x = self.upsample2(x)
        x = self.upconv2(x)
        x = self.upnorm2(x)

        #layer 1 up:
        x = self.uprelu1(torch.cat([x1, x], 1))
        x = self.upsample1(x)
        x = self.upconv1(x)
        
        return x, x_latent
