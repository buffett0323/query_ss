import torch
import torch.nn as nn
import torch.nn.functional as F
import math
try:
    from .util import (
        repeat_adsr_by_onset,
        build_phase_grid,
        gather_notes_pad,
        resample_adsr,
        sequencer
    )
except ImportError:
    # Fallback for when running the script directly
    from util import (
        repeat_adsr_by_onset,
        build_phase_grid,
        gather_notes_pad,
        resample_adsr,
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



# Phase-Align One-Shot ADSR Encoder
class ADSREncoderV3(nn.Module):
    def __init__(self,
                 proto_len: int = 87,
                 channels: int = 64,
                 hop: int = 512):
        super().__init__()
        self.proto_len = proto_len
        self.hop = hop
        self.eps = 1.0e-7


        self.backbone = DSBackbone(channels)

        # Attention pooling to single proto_E
        self.attn_pool = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, 1)
        )

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

        B, _, _ = log_rms.shape

        # 1) Get on-set index list
        on_idx = [torch.where(onset_flags[b, 0] == 1)[0].tolist() for b in range(B)]

        # 2) DSConv backbone
        adsr_feat = self.backbone(log_rms)  # (B,64,T)

        # 3) Zero-pad per note  →    note_E, mask
        note_E, mask = gather_notes_pad(
            adsr_feat, on_idx, self.proto_len)             # (B,N,64,L), (B,N,L)

        # 4) Attention or length-weighted averaging
        if note_E.size(1) > 0:
            # Mask-based averaging: normalize each note by valid length
            lengths = mask.sum(-1, keepdim=True)                    # (B,N,1)
            w = lengths / (lengths.sum(1, keepdim=True) + self.eps) # (B,N,1)
            w = w.unsqueeze(-1)                                     # (B,N,1,1)
            proto_E = (note_E * w).sum(dim=1)                       # (B,64,L)
        else:
            proto_E = note_E.mean(dim=1)

        # 5) Sequencer
        adsr_stream = sequencer(proto_E, onset_flags)            # (B,64,T)

        return {"proto_E": proto_E,
                "adsr_stream": adsr_stream}



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

    # # Create dummy ADSR embedding
    # tmp_len = 24
    # adsr_embed = torch.zeros(1, 64, 87)
    # adsr_embed[:, :, 0] = 99 * torch.ones(1, 64)
    # adsr_embed[:, :, 1:tmp_len+1] = torch.randn(1, 64, tmp_len)

    # print(f"Onset shape: {onset.shape}")
    # print(f"ADSR embed shape: {adsr_embed.shape}")
    # print(f"Onset positions: {torch.where(onset[0, 0] == 1)[0].tolist()}")

    # # Test the function
    # result = repeat_adsr_by_onset(adsr_embed, onset)
    # print(f"Result shape: {result.shape}")

    # # Verify the concept: check that ADSR is applied correctly
    # onset_positions = torch.where(onset[0, 0] == 1)[0]
    # print(f"\nOnset positions: {onset_positions.tolist()}")

    # for i, start_pos in enumerate(onset_positions):
    #     if i + 1 < len(onset_positions):
    #         end_pos = onset_positions[i + 1]
    #     else:
    #         end_pos = onset.shape[-1]

    #     segment_length = end_pos - start_pos
    #     print(f"Segment {i}: position {start_pos} to {end_pos} (length: {segment_length})")

    #     # Check that the segment is not all zeros (ADSR was applied)
    #     segment_data = result[0, :, start_pos:end_pos]
    #     non_zero_count = (segment_data != 0).sum().item()
    #     print(f"  Non-zero elements in segment: {non_zero_count}/{segment_data.numel()}")

    # print(result[:,:,0])
    # print(result[:,:,1])
    # print(result[:,:,43])
    # print(result[:,:,44])

    # print("\nFunction test completed!")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = ADSREncoderV3(proto_len=87, channels=64).to(device)

    p_onset = onset.to(device)
    wav = torch.randn(1, 1, 44100).to(device)

    out = model(wav, p_onset)
    print(out["proto_E"].shape)     # (1, 64, 87)
    print(out["adsr_stream"].shape) # (1, 64, 87)
