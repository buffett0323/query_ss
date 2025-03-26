# Buffett added
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from torch.cuda.amp import autocast

import typing as tp
from model.base import BaseModule
from model.diffusion import Diffusion
from model.styleencoder import StyleEncoder
from model.pitchencoder import PitchEncoder
from model.utils import sequence_mask, fix_len_compatibility

import utils
import transformers 
import commons
from modules_sf.modules import *
from commons import init_weights, get_padding  
from model.basic_pitch_encoder import init_basic_pitch_model, basic_pitch_encoder
from model.timbre_encoder import init_timbre_encoder, timbre_encoder  


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 mel_size=80,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, mel_size, 1)

    def forward(self, x, x_mask, g=None):
        x = self.pre(x * x_mask) * x_mask
        x = self.enc(x, x_mask, g=g)
        x = self.proj(x) * x_mask

        return x


class SynthesizerTrn(nn.Module):
    def __init__(self,
                 hps,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 encoder_hidden_size,
                 **kwargs):
        super().__init__()
        
        self.hps = hps
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size

        # Speaker Representation -- Timbre Encoder: Simsiam inference
        # self.emb_g = init_timbre_encoder(path=self.hps.data.timbre_encoder_path)
        self.emb_g = StyleEncoder(in_dim=80, hidden_dim=256, out_dim=256)
        
        # Pitch Encoder: Basic Pitch yn
        self.emb_p = init_basic_pitch_model()

        # Decoder
        self.dec_s = Decoder(encoder_hidden_size, encoder_hidden_size, 5, 1, 8, mel_size=80, gin_channels=hps.model.gin_channels) # 256 or 2048

    
    def forward(self, x, x_mel, mel_fn_x, length, mixup=False, time_dim=200):
        # Basic Pitch Encoder
        f0 = self.emb_p(x)['note'].permute(0, 2, 1)# torch.Size([4, 251, 88]) -> [4, 88, 251]
        f0 = F.interpolate(f0, size=time_dim, mode="nearest") # [4, 88, 251] -> [4, 88, 200]
        
        # Timbre Encoder
        x_mask = torch.unsqueeze(commons.sequence_mask(length, mel_fn_x.size(2)), 1).to(mel_fn_x.dtype)
        g = self.emb_g(mel_fn_x, x_mask).unsqueeze(-1)

        # Mix-up Training
        if mixup is True:
            g_mixup = torch.cat([g, g[torch.randperm(g.size()[0])]], dim=0)
            x_mask = torch.cat([x_mask, x_mask], dim=0)
            f0 = torch.cat([f0, f0], dim=0)
                        
            y_s = self.dec_s(f0, x_mask, g=g_mixup)
        else:
            y_s = self.dec_s(f0, x_mask, g=g)

        return g, y_s
        
    
    def voice_conversion(self, x, x_mel, mel_fn_x, x_length, y, y_mel, mel_fn_y, y_length, time_dim=200):
        # Get x's pitch (w2v, f0_code are x's features)
        f0 = self.emb_p(x)['note'].permute(0, 2, 1) #, x_mask)#.unsqueeze(-1) # V
        # f0 = self.proj_dense(f0) # torch.Size([4, 200, 88]) -> [4, 88, 251] -> [4, 88, 200]
        f0 = F.interpolate(f0, size=time_dim, mode="nearest") # [4, 88, 251] -> [4, 88, 200]
        
        x_mask = torch.unsqueeze(commons.sequence_mask(x_length, f0.size(2)), 1).to(x_mel.dtype) # V
        
        # Get y's timbre
        y_mask = torch.unsqueeze(commons.sequence_mask(y_length, y_mel.size(2)), 1).to(y_mel.dtype)
        g_from_y = self.emb_g(y_mel).unsqueeze(-1)

        # Decoder
        o_s = self.dec_s(f0, x_mask, g=g_from_y)
        
        return g_from_y, o_s #spk, src_out




class DDDM(BaseModule):
    def __init__(self, n_feats, spk_dim, dec_dim, beta_min, beta_max, hps):
        super(DDDM, self).__init__()  
        self.n_feats = n_feats
        self.spk_dim = spk_dim
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.encoder = SynthesizerTrn(
            hps,
            hps.data.n_mel_channels,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model
        )
        self.decoder = Diffusion(n_feats, dec_dim, spk_dim, beta_min, beta_max)

    @torch.no_grad() # For evaluation
    def forward(self, x, mel_x, mel_fn_x, x_lengths, n_timesteps, mode='ml'): 
        x_mask = sequence_mask(x_lengths, mel_fn_x.size(2)).unsqueeze(1).to(mel_fn_x.dtype) 
        
        spk, src_out = self.encoder(x, mel_x, mel_fn_x, x_lengths)
        src_mean_x = self.decoder.compute_diffused_mean(mel_fn_x, x_mask, src_out, 1.0)

        b = mel_fn_x.shape[0]
        max_length = int(x_lengths.max())
        max_length_new = fix_len_compatibility(max_length)
        x_mask_new = sequence_mask(x_lengths, max_length_new).unsqueeze(1).to(mel_fn_x.dtype)
        
        src_new = torch.zeros((b, self.n_feats, max_length_new), dtype=mel_fn_x.dtype, device=mel_fn_x.device)
        src_x_new = torch.zeros((b, self.n_feats, max_length_new), dtype=mel_fn_x.dtype, device=mel_fn_x.device)

        for i in range(b):
            src_new[i, :, :x_lengths[i]] = src_out[i, :, :x_lengths[i]]
            src_x_new[i, :, :x_lengths[i]] = src_mean_x[i, :, :x_lengths[i]]

        z_src = src_x_new
        z_src += torch.randn_like(src_x_new, device=src_x_new.device)

        y = self.decoder(z_src, x_mask_new, src_new, spk, n_timesteps, mode)
        enc_out = src_out # + ftr_out
        
        return enc_out, y[:, :, :max_length]
    
    def vc(self, x, mel_x, mel_fn_x, x_lengths, y, mel_y, mel_fn_y, y_lengths, n_timesteps, mode='ml'): 
        x_mask = sequence_mask(x_lengths, mel_fn_x.size(2)).unsqueeze(1).to(mel_fn_x.dtype)

        spk, src_out = self.encoder.voice_conversion(x, mel_x, mel_fn_x, x_lengths, y, mel_y, mel_fn_y, y_lengths)
        src_mean_x = self.decoder.compute_diffused_mean(mel_fn_x, x_mask, src_out, 1.0)

        b = mel_fn_x.shape[0]
        max_length = int(x_lengths.max())
        max_length_new = fix_len_compatibility(max_length)
        x_mask_new = sequence_mask(x_lengths, max_length_new).unsqueeze(1).to(mel_fn_x.dtype)
        src_new = torch.zeros((b, self.n_feats, max_length_new), dtype=mel_fn_x.dtype, device=mel_fn_x.device)
        src_x_new = torch.zeros((b, self.n_feats, max_length_new), dtype=mel_fn_x.dtype, device=mel_fn_x.device)

        for i in range(b):
            src_new[i, :, :x_lengths[i]] = src_out[i, :, :x_lengths[i]]
            src_x_new[i, :, :x_lengths[i]] = src_mean_x[i, :, :x_lengths[i]]

        z_src = src_x_new
        z_src += torch.randn_like(src_x_new, device=src_x_new.device)

        y = self.decoder(z_src, x_mask_new, src_new, spk, n_timesteps, mode)

        return y[:, :, :max_length]
    
    def compute_loss(self, x, mel_x, mel_fn_x, x_length): 
        # Encoder
        x_mask = sequence_mask(x_length, mel_fn_x.size(2)).unsqueeze(1).to(mel_fn_x.dtype)
        spk, src_out = self.encoder(x, mel_x, mel_fn_x, x_length, mixup=True, time_dim=mel_fn_x.size(2))

        # Mix-up
        mixup = torch.randint(0, 2, (mel_fn_x.size(0), 1, 1)).to(mel_fn_x.device)
        src_out_new = mixup*src_out[:mel_fn_x.size(0), :, :] + (1-mixup)*src_out[mel_fn_x.size(0):, :, :]
    
        # Decoder of DDDM
        diff_loss = self.decoder.compute_loss(mel_fn_x, x_mask, src_out_new, spk)
        enc_out = src_out[:mel_fn_x.size(0), :, :] #+ ftr_out[:x.size(0), :, :]
        mel_loss = F.l1_loss(mel_fn_x, enc_out)

        return diff_loss, mel_loss
