import math
import librosa
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn import init
from torch.nn.parameter import Parameter
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from models.e2e.bandit.bandsplit import BandSplitModule
from models.e2e.bandit.utils import MusicalBandsplitSpecification
from models.e2e.bandit.maskestim import OverlappingMaskEstimationModule
from models.e2e.bandit.tfmodel import SeqBandModellingModule
from models.e2e.querier.passt import Passt, PasstWrapper
from models.e2e.conditioners.film import FiLM
from models.types import InputType, OperationMode, SimpleishNamespace
from beats.BEATs import BEATs, BEATsConfig

from unet import UnetIquery
from functools import partial
from mamba_ssm import Mamba
from mamba.mamba_ssm.modules.mamba2 import Mamba2
from mamba.mamba_ssm.modules.mamba_simple import Mamba
from mamba.mamba_ssm.modules.block import Block
from mamba.mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba.mamba_ssm.ops.triton.layer_norm import RMSNorm


if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)

class BaseEndToEndModule(pl.LightningModule):

    def __init__(
        self,
    ) -> None:
        super().__init__()
        
        
class MyBandSplit(BaseEndToEndModule):
    def __init__(
        self,
        in_channel: int,
        stems: List[str],
        band_type: str = "musical",
        n_bands: int = 64,
        additive_film: bool = True,
        multiplicative_film: bool = True,
        film_depth: int = 2,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
        n_sqm_modules: int = 12,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
        mlp_dim: int = 512,
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: Dict | None = None,
        complex_mask: bool = True,
        use_freq_weights: bool = True,
        n_fft: int = 2048,
        win_length: int | None = 2048,
        hop_length: int = 512,
        window_fn: str = "hann_window",
        wkwargs: Dict | None = None,
        power: int | None = None,
        center: bool = True,
        normalized: bool = True,
        pad_mode: str = "constant",
        onesided: bool = True,
        use_beats: bool = True, # False
        fs: int = 44100,
    ):
        super().__init__()
        
        
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
        
        
        self.instantiate_bandsplit(
            in_channel=in_channel,
            band_type=band_type,
            n_bands=n_bands,
            require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap,
            normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature,
            emb_dim=emb_dim,
            n_fft=n_fft,
            fs=fs,
        )
        
        
        self.instantiate_tf_modelling(
            n_sqm_modules=n_sqm_modules,
            emb_dim=emb_dim,
            rnn_dim=rnn_dim,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
        )
        

        self.instantiate_mask_estim(
            in_channel=in_channel,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            n_freq=n_fft // 2 + 1,
            use_freq_weights=use_freq_weights,
        )
        

        self.query_encoder = Passt(
            original_fs=fs,
            passt_fs=32000,
        )
        

        self.film = FiLM(
            self.query_encoder.PASST_EMB_DIM,
            emb_dim,
            additive=additive_film,
            multiplicative=multiplicative_film,
            depth=film_depth,
        )
        
        self.use_beats = use_beats
        if self.use_beats:
            self.instantiate_beats(beats_check_point_pth='beats/pt_dict/BEATs_iter3_plus_AS2M.pt')

    def instantiate_beats(
        self,
        beats_check_point_pth='beats/pt_dict/BEATs_iter3_plus_AS2M.pt',
    ):
        checkpoint = torch.load(beats_check_point_pth)

        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint['model'])
        self.beats = BEATs_model.eval().cuda()


    def instantiate_bandsplit(
        self,
        in_channel: int,
        band_type: str = "musical",
        n_bands: int = 64,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
        emb_dim: int = 128,
        n_fft: int = 2048,
        fs: int = 44100,
    ):

        assert band_type == "musical"

        self.band_specs = MusicalBandsplitSpecification(
            nfft=n_fft, fs=fs, n_bands=n_bands
        )

        self.band_split = BandSplitModule(
            in_channel=in_channel,
            band_specs=self.band_specs.get_band_specs(),
            require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap,
            normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature,
            emb_dim=emb_dim,
        )
        
        
        
    def beats_query(self, wav):
        wav = wav.mean(dim=1, keepdim=False)
        padding_mask = torch.zeros(1, wav.shape[1]).bool().cuda()  # Move padding mask to GPU
        embed = []

        for i in range(wav.shape[0]):
            embed.append(self.beats.extract_features(wav[i].unsqueeze(0), padding_mask=padding_mask)[0].mean(dim=1, keepdim=False))

        embed = torch.cat(embed, dim=0)
        return embed
    
    
    
    def instantiate_tf_modelling(
        self,
        n_sqm_modules: int = 12,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
    ):
        self.tf_model = SeqBandModellingModule(
            n_modules=n_sqm_modules,
            emb_dim=emb_dim,
            rnn_dim=rnn_dim,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
        )
        
        
    def instantiate_mask_estim(
        self,
        in_channel: int,
        emb_dim: int,
        mlp_dim: int,
        hidden_activation: str,
        hidden_activation_kwargs: Optional[Dict] = None,
        complex_mask: bool = True,
        n_freq: Optional[int] = None,
        use_freq_weights: bool = True,
    ):
        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}

        assert n_freq is not None

        self.mask_estim = OverlappingMaskEstimationModule(
            band_specs=self.band_specs.get_band_specs(),
            freq_weights=self.band_specs.get_freq_weights(),
            n_freq=n_freq,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            in_channel=in_channel,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            use_freq_weights=use_freq_weights,
        )
        
        
    def mask(self, x, m):
        return x * m
    
    
    def forward(self, batch: InputType):
        with torch.no_grad():
            x = self.stft(batch.mixture.audio)
            batch.mixture.spectrogram = x

            if "sources" in batch.keys():
                for stem in batch.sources.keys():
                    s = batch.sources[stem].audio
                    s = self.stft(s)
                    batch.sources[stem].spectrogram = s
                    
        batch = self.separate(batch)

        return batch
    
    
    def separate(self, batch):

        x, q, length = self.encode(batch)

        q = self.adapt_query(q, batch)

        m = self.mask_estim(q)
        s = self.mask(x, m)
        s = torch.reshape(s, x.shape)
        batch.estimates["target"] = SimpleishNamespace(
            audio=self.istft(s, length), spectrogram=s
        )

        return batch
    
    
    def encode(self, batch):
        x = batch.mixture.spectrogram
        length = batch.mixture.audio.shape[-1]
        z = self.band_split(x)  # (batch, emb_dim, n_band, n_time)
        q = self.tf_model(z)  # (batch, emb_dim, n_band, n_time)

        return x, q, length
    
    
    def adapt_query(self, q, batch):
        if self.use_beats:
            w = self.beats_query(batch.query.audio)
        else:
            w = self.query_encoder(batch.query.audio)
            
        # BF: torch.Size([1, 2, 441000]) 
        # AF: torch.Size([1, 768])
        q = torch.permute(q, (0, 3, 1, 2)) # (batch, n_band, n_time, emb_dim) -> (batch, emb_dim, n_band, n_time)
        q = self.film(q, w)
        q = torch.permute(q, (0, 2, 3, 1)) # -> (batch, n_band, n_time, emb_dim)
        
        return q
    
###------------------------------------------------------------------------------------------###

class BandSplitMamba(BaseEndToEndModule):
    def __init__(
        self,
        in_channel: int,
        stems: List[str],
        band_type: str = "musical",
        n_bands: int = 64,
        additive_film: bool = True,
        multiplicative_film: bool = True,
        film_depth: int = 2,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
        n_sqm_modules: int = 12,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
        mlp_dim: int = 512,
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: Dict | None = None,
        complex_mask: bool = True,
        use_freq_weights: bool = True,
        n_fft: int = 2048,
        win_length: int | None = 2048,
        hop_length: int = 512,
        window_fn: str = "hann_window",
        wkwargs: Dict | None = None,
        power: int | None = None,
        center: bool = True,
        normalized: bool = True,
        pad_mode: str = "constant",
        onesided: bool = True,
        use_beats: bool = True, # False
        fs: int = 44100,
    ):
        super().__init__()
        
        
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
        
        
        self.instantiate_bandsplit(
            in_channel=in_channel,
            band_type=band_type,
            n_bands=n_bands,
            require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap,
            normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature,
            emb_dim=emb_dim,
            n_fft=n_fft,
            fs=fs,
        )


        self.instantiate_mask_estim(
            in_channel=in_channel,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            n_freq=n_fft // 2 + 1,
            use_freq_weights=use_freq_weights,
        )
        

        self.query_encoder = Passt(
            original_fs=fs,
            passt_fs=32000,
        )
        

        self.film = FiLM(
            self.query_encoder.PASST_EMB_DIM,
            emb_dim,
            additive=additive_film,
            multiplicative=multiplicative_film,
            depth=film_depth,
        )

        self.instantiate_mamba(
            emb_dim=64, # 48,
            eps=1.0e-5,
            emb_ks=4,
            emb_hs=1,
            n_layer=6, #6,
            bidirectional=True, #True,
        )
        
        
    def instantiate_mamba(
        self,
        emb_dim=64, # 48,
        eps=1.0e-5,
        emb_ks=4,
        emb_hs=1,
        n_layer=6,
        bidirectional=True,
    ):
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        in_channels = emb_dim * emb_ks

        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_mamba = MambaBlock(in_channels=emb_dim, n_layer=n_layer, bidirectional=bidirectional)
        self.intra_linear = nn.Conv1d(
            in_channels=128,  # Input channels (C)
            out_channels=128,  # Output channels (same as input in this case)
            kernel_size=4,
            stride=2,
            padding=1
        )
        # nn.ConvTranspose1d(
        #     in_channels * 2, emb_dim, emb_ks, stride=emb_hs
        # )

    def instantiate_bandsplit(
        self,
        in_channel: int,
        band_type: str = "musical",
        n_bands: int = 64,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
        emb_dim: int = 128,
        n_fft: int = 2048,
        fs: int = 44100,
    ):

        assert band_type == "musical"

        self.band_specs = MusicalBandsplitSpecification(
            nfft=n_fft, fs=fs, n_bands=n_bands
        )

        self.band_split = BandSplitModule(
            in_channel=in_channel,
            band_specs=self.band_specs.get_band_specs(),
            require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap,
            normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature,
            emb_dim=emb_dim,
        )
    
        
        
    def instantiate_mask_estim(
        self,
        in_channel: int,
        emb_dim: int,
        mlp_dim: int,
        hidden_activation: str,
        hidden_activation_kwargs: Optional[Dict] = None,
        complex_mask: bool = True,
        n_freq: Optional[int] = None,
        use_freq_weights: bool = True,
    ):
        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}

        assert n_freq is not None

        self.mask_estim = OverlappingMaskEstimationModule(
            band_specs=self.band_specs.get_band_specs(),
            freq_weights=self.band_specs.get_freq_weights(),
            n_freq=n_freq,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            in_channel=in_channel,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            use_freq_weights=use_freq_weights,
        )
        
        
    def mask(self, x, m):
        return x * m
    
    
    def forward(self, batch: InputType):
        with torch.no_grad():
            x = self.stft(batch.mixture.audio)
            batch.mixture.spectrogram = x

            if "sources" in batch.keys():
                for stem in batch.sources.keys():
                    s = batch.sources[stem].audio
                    s = self.stft(s)
                    batch.sources[stem].spectrogram = s
                    
        batch = self.separate(batch)

        return batch
    
    
    def separate(self, batch):

        x, q, length = self.encode(batch)

        q = self.adapt_query(q, batch)

        m = self.mask_estim(q)
        s = self.mask(x, m)
        s = torch.reshape(s, x.shape)
        batch.estimates["target"] = SimpleishNamespace(
            audio=self.istft(s, length), spectrogram=s
        )

        return batch
    
    
    def encode(self, batch):
        x = batch.mixture.spectrogram
        length = batch.mixture.audio.shape[-1]
        z = self.band_split(x)  # (batch, emb_dim, n_band, n_time) # print(z.shape) # torch.Size([BS, 64, 576, 128])
        
        # Mamba Block
        z = self.adapt_mamba(z) # (batch, emb_dim, n_band, n_time): Same size

        return x, z, length
    
    
    def adapt_query(self, q, batch):
        w = self.query_encoder(batch.query.audio)
            
        # BF: torch.Size([1, 2, 441000]) 
        # AF: torch.Size([1, 768])
        q = torch.permute(q, (0, 3, 1, 2)) # (batch, n_band, n_time, emb_dim) -> (batch, emb_dim, n_band, n_time)
        q = self.film(q, w)
        q = torch.permute(q, (0, 2, 3, 1)) # -> (batch, n_band, n_time, emb_dim)
        
        return q
    

    def adapt_mamba(self, x):
        B, C, n_band, T = x.shape

        # No need for padding if Mamba2 can handle arbitrary sequence lengths
        # Apply normalization
        input_ = x  # Shape: [B, C, n_band, T] ([BS, 64, 576, 128])
        intra_rnn = self.intra_norm(input_)  # Shape: [B, C, n_band, T]

        # Rearrange the tensor to merge batch and n_band dimensions
        intra_rnn = intra_rnn.permute(0, 2, 1, 3).contiguous()  # [B, n_band, C, T]
        intra_rnn = intra_rnn.view(B * n_band, C, T)  # [B * n_band, C, T]

        # Transpose to get the sequence dimension first
        intra_rnn = intra_rnn.transpose(1, 2)  # [B * n_band, T, C]

        # Apply the Mamba block over the time dimension
        # print("Before mamba:", intra_rnn.shape) # torch.Size([1152, 128, 64]) 
        intra_rnn = self.intra_mamba(intra_rnn)  # [B * n_band, T, C] ([1152, 128, 64])
        # print("After mamba:", intra_rnn.shape) # Bidirection: torch.Size([1152, 128, 128])
        
        
        # Transpose back to [B * n_band, C, T]
        intra_rnn = intra_rnn.transpose(1, 2)  # [B * n_band, C, 2*T]
        
        # Intra Linear Conv1D
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
        # print("After Linear:", intra_rnn.shape) # After Linear: torch.Size([1152, 128, 64])
        
        # Reshape back to original dimensions
        intra_rnn = intra_rnn.view(B, n_band, C, T)  # [B, n_band, C, T]

        # Permute to get back to [B, C, n_band, T]
        intra_rnn = intra_rnn.permute(0, 2, 1, 3).contiguous()  # [B, C, n_band, T]

        # Add the residual connection
        intra_rnn = intra_rnn + input_  # [B, C, n_band, T]

        return intra_rnn

    
    
    
class MambaBlock(nn.Module):
    def __init__(self, in_channels, n_layer=1, bidirectional=False):
        super(MambaBlock, self).__init__()
        self.forward_blocks = nn.ModuleList()
        for i in range(n_layer):
            self.forward_blocks.append(
                Block(
                    dim=in_channels,
                    mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=4),
                    mlp_cls=nn.Identity,  # Assuming no additional MLP layer
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=False,
                )
            )
        if bidirectional:
            self.backward_blocks = nn.ModuleList()
            for i in range(n_layer):
                self.backward_blocks.append(
                    Block(
                        dim=in_channels,
                        mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=4),
                        mlp_cls=nn.Identity,
                        norm_cls=partial(RMSNorm, eps=1e-5),
                        fused_add_norm=False,
                    )
                )
        else:
            self.backward_blocks = None

        self.apply(partial(_init_weights, n_layer=n_layer))



    def forward(self, input):
        # Forward pass through the forward_blocks
        residual = None
        forward_f = input
        for block in self.forward_blocks:
            forward_f, residual = block(forward_f, residual, inference_params=None)
        output = forward_f  # The residual is handled within the Block class

        # If bidirectional, process the sequence in reverse
        if self.backward_blocks is not None:
            backward_f = torch.flip(input, dims=[1])  # Flip along the time dimension
            back_residual = None
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=None)
            back_output = backward_f
            back_output = torch.flip(back_output, dims=[1])
            output = torch.cat([output, back_output], dim=-1)  # Concatenate along the feature dimension

        return output
    
    
    
class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            _, C, _, _ = x.shape
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat