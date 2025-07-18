import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import math
try:
    from .util import (
        gather_notes_pad,
        sequencer
    )
except ImportError:
    from util import (
        gather_notes_pad,
        sequencer
    )


# ADSR Encoder V1
class ResidualDilatedBlock(nn.Module):
    """
    Dilated 1-D residual block (kernel=3) with weight-norm Conv.
    """
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(channels, channels, kernel_size=3,
                      padding=dilation, dilation=dilation))
        self.act  = nn.GELU()

    def forward(self, x):
        return x + self.act(self.conv(x))


# ADSR Encoder V2, V3: Depthwise-separable convolution
class DSConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, d: int):
        super().__init__()
        pad = (k - 1) // 2 * d
        self.depthwise = nn.Conv1d(in_ch, in_ch, k,
                                   groups=in_ch, dilation=d,
                                   padding=pad, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act  = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


# ADSR Encoder V2, V3: Basic convolution backbone
class ConvBackbone(nn.Module):
    """Convert log-RMS to amplitude features"""
    def __init__(self, ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.GELU(),
            nn.Conv1d(32, 64, 3, padding=1, dilation=2), nn.GELU(),
            nn.Conv1d(64, ch, 3, padding=1, dilation=4), nn.GELU(),
            nn.BatchNorm1d(ch)
        )

    def forward(self, x):
        return self.net(x)          # (B,64,T)


# ADSR Encoder V3: Depthwise-separable backbone
class DSBackbone(nn.Module):
    def __init__(self, ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.GELU(),          # early channel mix
            DSConv1d(32, 64, k=3, d=1),
            DSConv1d(64, 64, k=3, d=2),
            DSConv1d(64, 64, k=3, d=4),
            DSConv1d(64, 64, k=3, d=8),
        )

    def forward(self, x):
        return self.net(x)          # (B, 64, T)


# ADSR Encoder V2
class ResidualTCN(nn.Module):
    """
    Dilated residual temporal convolution block.

    Architecture:
    x --(depthwise dilated conv)--> BN/GELU --> 1×1 conv --+
      |                                                    |
      +---------------------------(residual add)-----------+

    Args:
        channels: Number of input/output channels
        k: Kernel size (usually 3)
        d: Dilation factor (2^i)
    """
    def __init__(self, channels: int, k: int = 3, d: int = 1):
        super().__init__()
        pad = (k - 1) * d                       # Maintain length

        # Depth-wise dilated conv (groups=channels)
        self.depthwise = nn.utils.weight_norm(
            nn.Conv1d(channels, channels, kernel_size=k,
                      dilation=d, padding=pad,
                      groups=channels, bias=False)
        )
        # Point-wise conv 1×1
        self.pointwise = nn.utils.weight_norm(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        )

        self.norm = nn.BatchNorm1d(channels)
        self.act  = nn.GELU()

    def forward(self, x):
        """
        x : (B, channels, T)
        """
        y = self.depthwise(x)
        y = self.pointwise(self.act(self.norm(y)))
        # Trim right side if padding causes length > T
        if y.size(-1) != x.size(-1):
            y = y[..., :x.size(-1)]
        return x + y


class ADSREncoderV1(nn.Module):
    """
    Temporal multi-scale ConvNet + BiLSTM.
    Input  : waveform  (B, 1, 44100)
    Output : ADSR embedding  (B, 64, 87)   # 44100 / 512 ≈ 86.1 ≈ 87
    Waveform (44100 samples)
        ↓
    Frame-based features (87 frames)
        ↓
    Multi-scale temporal modeling
        ↓
    Bidirectional sequence processing
        ↓
    ADSR embedding (64 dims × 87 frames)
    """
    def __init__(self,
                 hop: int = 512,
                 embed_channels: int = 64,
                 dilations=(1, 2, 4, 8, 16),
                 lstm_layers: int = 2,
                 lstm_hidden: int = 32):
        super().__init__()
        self.hop = hop                        # frame hop in samples
        self.eps = 1.0e-7

        # Envelope pre-processor
        # 1×1 conv (maps [log-RMS, Δlog-RMS] → C₀)
        self.pre = nn.Conv1d(2, embed_channels, kernel_size=1)

        # Dilated residual stack
        self.dilated = nn.ModuleList(
            [ResidualDilatedBlock(embed_channels, d) for d in dilations])

        # Low-rate context branch (4x downsample for global context)
        self.lowrate = nn.Sequential(
            nn.Conv1d(embed_channels, embed_channels, kernel_size=3,
                      padding=1, stride=4),
            nn.GELU())

        # BiLSTM over concatenated high+low streams
        self.bilstm = nn.LSTM(
            input_size=embed_channels * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True)

        # 1×1 conv to final 64-d embedding
        self.out_proj = nn.Conv1d(lstm_hidden * 2, embed_channels, 1)

    def preprocess(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (B, 1, T_samples) in [-1,1]
        returns (B, 1, T_frames)
        """
        length = wav.shape[-1]
        right_pad = math.ceil(length / self.hop) * self.hop - length
        wav = nn.functional.pad(wav, (0, right_pad))
        return wav


    # --------------------------------------------------------------------- #
    #  helper: frame-wise log-RMS + derivative
    # --------------------------------------------------------------------- #
    def _envelope_features(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (B, 1, T_samples) in [-1,1]
        returns (B, 2, T_frames) : [log-RMS, Δlog-RMS]
        """
        # square + pool = frame energy
        rms = torch.sqrt(
            F.avg_pool1d(wav ** 2, kernel_size=self.hop, stride=self.hop) + self.eps
        )                                       # (B,1,Tf)

        log_rms = torch.log(rms + self.eps)          # (B,1,Tf)
        # first-order diff (pad with zero at t=0)
        diff = torch.cat([log_rms[:, :, :1],
                          log_rms[:, :, 1:] - log_rms[:, :, :-1]], dim=2)
        feats = torch.cat([log_rms, diff], dim=1)   # (B,2,Tf)
        return feats


    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # 0) preprocess
        wav = self.preprocess(wav)

        # 1) envelope features
        feats = self._envelope_features(wav)    # (B,2,Tf)

        # 2) initial projection
        x = self.pre(feats)                     # (B,C,Tf)

        # 3) dilated residual stack
        for block in self.dilated:
            x = block(x)                        # (B,C,Tf)

        # 4) Multi-scale concatenation (high-rate + up-sampled low-rate)
        low = self.lowrate(x)                   # (B,C,Tf/4)
        low = F.interpolate(low, size=x.shape[-1],
                            mode='linear', align_corners=False)
        x_cat = torch.cat([x, low], dim=1)      # (B,2C,Tf)

        # 5) BiLSTM (time dimension first for batch-first=True)
        x_cat = x_cat.transpose(1, 2)           # (B,Tf,2C)
        lstm_out, _ = self.bilstm(x_cat)        # (B,Tf,2*hidden)
        lstm_out = lstm_out.transpose(1, 2)     # (B,2*hidden,Tf)

        # 6) final projection → (B,64,Tf)
        z_a = self.out_proj(lstm_out)
        return z_a



class ADSREncoderV2(nn.Module):
    def __init__(self, proto_len=87, ch=64):
        super().__init__()
        # ---------- OnsetNet (can be replaced with pre-trained madmom CNN) ----------
        # self.onset_net = CNNOnsetNet()  # Any network outputting (B,1,T) ∈ (0,1)

        # ---------- Context Encoder ----------
        self.ctx = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.GELU(),
            DSConv1d(32, 64, k=3, d=2), nn.GELU(),
            DSConv1d(64, 64, k=3, d=4), nn.GELU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc_ctx  = nn.Linear(64, 128)

        # ---------- FiLM generator ----------
        self.film = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(),
            nn.Linear(256, ch * 2 * 16))  # 16 TCN layers, each with γ,β

        # ---------- One-Shot TCN ----------
        self.tcn = nn.ModuleList(
            [ResidualTCN(ch, k=3, d=2**i) for i in range(16)]
        )
        self.proto_len = proto_len
        self.ch = ch

    # ----------------------------------------------------------

    def forward(self, p_onset, log_rms):
        """
        Args:
            p_onset: (B,1,T) – Onset probability (can be pre-computed)
            log_rms: (B,1,T) – Log-RMS features (pre-computed or computed on-the-fly)
        """
        # 1) Onset detection (soft probability)
        # p_onset = self.onset_net(wav)            # (B,1,T) ∈ (0,1)
        # delta   = (p_onset > 0.5).float()        # For fully differentiable, use p_onset directly

        # 2) Context embedding
        e = self.ctx(log_rms).squeeze(-1)        # (B,64)
        cond = self.fc_ctx(e)                    # (B,128)
        film = self.film(cond).view(-1, 16, 2, self.ch, 1)

        # 3) Synth proto ADSR
        x = torch.randn(log_rms.size(0), 1, self.proto_len,
                        device=log_rms.device)

        for i, block in enumerate(self.tcn):
            γ, β = film[:, i, 0], film[:, i, 1]
            x = block(x)
            x = γ * x + β
        proto_E = x                              # (B,64,L)

        # 4) Sequencer
        adsr_stream = sequencer(proto_E, p_onset)  # (B,64,T)

        return adsr_stream, proto_E, p_onset



# ADSR Encoder
class ADSREncoderV3(nn.Module):
    def __init__(self,
                 channels: int = 64,
                 hop: int = 512,
                 method: str = "length-weighted"):
        super().__init__()
        self.hop = hop
        self.eps = 1.0e-7

        self.backbone = DSBackbone(channels)
        self.method = method

    def preprocess(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (B, 1, T_samples) in [-1,1]
        returns (B, 1, T_frames)
        """
        length = wav.shape[-1]
        right_pad = math.ceil(length / self.hop) * self.hop - length
        wav = nn.functional.pad(wav, (0, right_pad))
        return wav

    def _envelope_features(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (B, 1, T_samples) in [-1,1]
        returns (B, 1, T_frames) : log-RMS
        """
        # square + pool = frame energy
        rms = torch.sqrt(
            F.avg_pool1d(wav ** 2, kernel_size=self.hop, stride=self.hop) + self.eps
        )                                       # (B,1,Tf)
        log_rms = torch.log(rms + self.eps)          # (B,1,Tf)
        return log_rms

    def forward(self,
                wav: torch.Tensor,          # (B,1,T_samples) waveform input
                onset_flags: torch.Tensor   # (B,1,T_frames)  0/1 impulse train
                ) -> dict:

        # 0) Preprocess waveform and calculate log-RMS
        wav = self.preprocess(wav)          # (B,1,T_samples)
        log_rms = self._envelope_features(wav)  # (B,1,T_frames)

        B, _, P = log_rms.shape

        # 1) Get on-set index list
        on_idx = [torch.where(onset_flags[b, 0] == 1)[0].tolist() for b in range(B)]

        # 2) DSConv backbone
        adsr_feat = self.backbone(log_rms)  # (B,64,T)

        # 3) Zero-pad per note  →    note_E, mask
        note_E, mask = gather_notes_pad(
            adsr_feat, on_idx, P)             # (B,N,64,L), (B,N,L)

        # 4) Length-weighted averaging
        if self.method == "length-weighted":
            # Mask-based averaging: normalize each note by valid length
            lengths = mask.sum(-1, keepdim=True)                    # (B,N,1)
            w = lengths / (lengths.sum(1, keepdim=True) + self.eps) # (B,N,1)
            w = w.unsqueeze(-1)                                     # (B,N,1,1)
            proto_E = (note_E * w).sum(dim=1)                       # (B,64,L)

        elif self.method == "equal-weighted":
            valid = mask.unsqueeze(2)                 # (B,N,1,L)
            sum_notes = (note_E * valid).sum(dim=1)   # (B,C,L)
            num_notes = valid.sum(dim=1).clamp(min=1) # (B,1,L)
            proto_E = sum_notes / num_notes

        else:
            raise ValueError(f"Invalid method: {self.method}")

        return proto_E




# FiLM Module (kept from original)
class ADSRFiLM(nn.Module):
    """
    FiLM-gates a content latent with an ADSR embedding.

        C_hat = C * (1 + γ) + β
    where γ, β are learned linear projections of A.

    A : (B, adsr_ch,  T)
    C : (B, cont_ch,  T)
    ----------------------------------------------------------
    """
    def __init__(self,
                 adsr_ch:   int = 64,
                 cont_ch:   int = 256,
                 hidden_ch: int = 128):
        super().__init__()

        # Conv stack: (A) -> 2*cont_ch channels  (γ‖β)
        self.to_film = nn.Sequential(
            nn.Conv1d(adsr_ch, hidden_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_ch, 2 * cont_ch, kernel_size=1)
        )

        # initialise to identity: γ=0, β=0
        nn.init.zeros_(self.to_film[-1].weight)
        nn.init.zeros_(self.to_film[-1].bias)

    def forward(self, A: torch.Tensor, C: torch.Tensor):
        """
        A : (B, adsr_ch,  T)
        C : (B, cont_ch,  T)
        --------------------------------------------------
        returns
        C_tilde : (B, cont_ch, T)  -- FiLM-modulated content
        """
        γβ = self.to_film(A)                # (B, 2·C, T)
        γ, β = γβ.chunk(2, dim=1)           # split along channel
        C_tilde = C * (1.0 + γ) + β
        return C_tilde


if __name__ == "__main__":

    # Your example onset flags
    onset = torch.tensor([
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    onset = onset.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 87)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = ADSREncoderV3(channels=64).to(device)

    p_onset = onset.to(device)
    wav, _ = torchaudio.load("/mnt/gestalt/home/buffett/EDM_FAC_LOG/0716_mn/sample_audio/iter_0/conv_both/1_ref.wav")
    wav = wav.unsqueeze(0)
    wav = wav.to(device)

    out = model(wav, p_onset)
    print(out.shape) # 1, 64, 87

    # # Visualize the ADSR embedding (64, 87)
    # import matplotlib.pyplot as plt
    # import numpy as np

    # # Extract the actual embedding (remove batch dimension)
    # adsr_embedding = out.squeeze(0).detach().cpu().numpy()  # Shape: (64, 87)

    # # Create a figure with multiple visualization methods
    # fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # fig.suptitle('ADSR Encoder Output Visualization (64 channels × 87 time steps)', fontsize=16)

    # # 1. Heatmap visualization
    # im1 = axes[0, 0].imshow(adsr_embedding, aspect='auto', cmap='viridis')
    # axes[0, 0].set_title('Heatmap: All 64 channels over time')
    # axes[0, 0].set_xlabel('Time steps (87)')
    # axes[0, 0].set_ylabel('Channels (64)')
    # plt.colorbar(im1, ax=axes[0, 0])

    # # 2. Mean activation over time
    # mean_over_time = np.mean(adsr_embedding, axis=0)
    # axes[0, 1].plot(mean_over_time, linewidth=2)
    # axes[0, 1].set_title('Mean activation across all channels')
    # axes[0, 1].set_xlabel('Time steps (87)')
    # axes[0, 1].set_ylabel('Mean activation')
    # axes[0, 1].grid(True, alpha=0.3)

    # # 3. Channel-wise statistics
    # channel_means = np.mean(adsr_embedding, axis=1)
    # channel_stds = np.std(adsr_embedding, axis=1)
    # x_pos = np.arange(64)
    # axes[1, 0].bar(x_pos, channel_means, alpha=0.7, label='Mean')
    # axes[1, 0].set_title('Mean activation per channel')
    # axes[1, 0].set_xlabel('Channel index (0-63)')
    # axes[1, 0].set_ylabel('Mean activation')
    # axes[1, 0].grid(True, alpha=0.3)

    # # 4. Sample individual channels
    # sample_channels = [0, 15, 31, 47]  # Sample 4 channels
    # for i, ch_idx in enumerate(sample_channels):
    #     axes[1, 1].plot(adsr_embedding[ch_idx], label=f'Channel {ch_idx}', alpha=0.8)
    # axes[1, 1].set_title('Sample individual channels')
    # axes[1, 1].set_xlabel('Time steps (87)')
    # axes[1, 1].set_ylabel('Activation')
    # axes[1, 1].legend()
    # axes[1, 1].grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.savefig('adsr_embedding_visualization.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # # Print some statistics
    # print(f"\nADSR Embedding Statistics:")
    # print(f"Shape: {adsr_embedding.shape}")
    # print(f"Min value: {adsr_embedding.min():.4f}")
    # print(f"Max value: {adsr_embedding.max():.4f}")
    # print(f"Mean value: {adsr_embedding.mean():.4f}")
    # print(f"Std value: {adsr_embedding.std():.4f}")

    # # Show correlation with onset positions
    # onset_positions = torch.where(p_onset[0, 0] == 1)[0].cpu().numpy()
    # print(f"\nOnset positions: {onset_positions}")

    # # Highlight onset positions in the visualization
    # if len(onset_positions) > 0:
    #     fig2, ax = plt.subplots(figsize=(12, 6))
    #     im = ax.imshow(adsr_embedding, aspect='auto', cmap='viridis')
    #     ax.set_title('ADSR Embedding with Onset Positions Highlighted')
    #     ax.set_xlabel('Time steps (87)')
    #     ax.set_ylabel('Channels (64)')

    #     # Mark onset positions with vertical lines
    #     for onset_pos in onset_positions:
    #         ax.axvline(x=onset_pos, color='red', linestyle='--', alpha=0.8, linewidth=2)

    #     plt.colorbar(im)
    #     plt.tight_layout()
    #     plt.savefig('adsr_embedding_with_onsets.png', dpi=300, bbox_inches='tight')
    #     plt.show()
