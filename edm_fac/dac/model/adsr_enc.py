import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute mean over `dim` while ignoring positions where mask == 0.

    Args
    ----
    x     : (B, C, T) or (B, N, C) tensor
    mask  : broadcastable to x; same shape except at `dim`
    dim   : dimension to reduce over

    Returns
    -------
    Tensor with `dim` removed.
    """
    mask = mask.to(dtype=x.dtype)
    total = (x * mask).sum(dim=dim)
    denom = mask.sum(dim=dim).clamp_min(1e-6)
    return total / denom


def l2_normalise(vec: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """L2-normalise the last dimension."""
    return vec / (vec.norm(p=2, dim=-1, keepdim=True) + eps)


class DSConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size,
                                   groups=in_ch, dilation=dilation,
                                   padding=pad, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act  = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


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

        # ───────────────────── envelope pre-processor ───────────────────── #
        # 1×1 conv (maps [log-RMS, Δlog-RMS] → C₀)
        self.pre = nn.Conv1d(2, embed_channels, kernel_size=1)

        # Dilated residual stack
        self.dilated = nn.ModuleList(
            [ResidualDilatedBlock(embed_channels, d) for d in dilations])

        # Low-rate context branch (×4 downsample → global context)
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

        # 4) multi-scale concat (high-rate || up-sampled low-rate)
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


# New Note-pooled ADSR Encoder
class ADSREncoderV2(nn.Module):
    """
    A note-pooled ADSR encoder that turns audio into ONE global envelope code z_a ∈ ℝ^(param_dim + residual_dim).

    Input  : waveform  (B, 1, T_samples) in [-1,1]
    Output : ADSR code (B, param_dim + residual_dim)

    Audio (T_samples)
        ↓
    Frame-based log-RMS features (T_frames)
        ↓
    Multi-scale temporal modeling
        ↓
    Note-pooled features
        ↓
    ADSR code (param_dim + residual_dim)

    Args
    ----
    hop           : frame hop in samples
    d_model       : channel width of internal representation
    n_layers      : number of DSConv blocks
    param_dim     : number of explicit ADSR parameters (default 4: A,D,R,S_lvl)
    residual_dim  : dimension of extra residual code concatenated to params
    """
    def __init__(self,
                 hop: int = 512,
                 d_model: int = 64,
                 n_layers: int = 3,
                 param_dim: int = 4,
                 residual_dim: int = 4):
        super().__init__()
        self.hop = hop                        # frame hop in samples
        self.eps = 1.0e-7

        layers = []
        in_ch = 1
        dilations = [2 ** i for i in range(n_layers)]
        for d in dilations:
            layers.append(DSConv1d(in_ch, d_model, kernel_size=5, dilation=d))
            in_ch = d_model
        self.feat_extractor = nn.Sequential(*layers)

        # Two heads
        self.param_head = nn.Linear(d_model, param_dim)
        self.resid_head = nn.Linear(d_model, residual_dim)

        # Store hyper-params
        self.param_dim = param_dim
        self.residual_dim = residual_dim
        self.out_dim = param_dim + residual_dim

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

    # --------------------------------------------------------------------- #

    def forward(self,
                wav: torch.Tensor,
                note_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        wav        : (B, 1, T_samples) – audio waveform in [-1,1]
        note_mask  : (B, N, T_frames)  – binary mask per note (optional)
                                     1 where the note is active, else 0.
                                     If None, global average pooling is used.

        Returns
        -------
        z_a        : (B, out_dim) – concatenated ADSR code
        param_vec  : (B, param_dim) – raw param outputs (before concat)
        """
        # 0) preprocess audio
        wav = self.preprocess(wav)

        # 1) extract log-RMS features
        log_rms = self._envelope_features(wav)    # (B,1,Tf)
        print(log_rms.shape)

        B, _, T = log_rms.shape
        feats = self.feat_extractor(log_rms)          # (B, D, T)

        # ------------------ note-pool → z_feats -------------------------- #
        if note_mask is not None:
            # Masked mean over time for each note -> (B, N, D)
            note_feats = masked_mean(
                feats.unsqueeze(1),  # (B, 1, D, T)
                note_mask.unsqueeze(2),  # (B, N, 1, T)
                dim=-1
            )  # → (B, N, D)

            # Pool notes into ONE vector (mean or max)
            z_feats = note_feats.mean(dim=1)          # (B, D)
        else:
            # Global average pool over time
            z_feats = feats.mean(dim=-1)              # (B, D)

        # ------------------ heads --------------------------------------- #
        raw_params = self.param_head(z_feats)         # (B, P)
        residual   = l2_normalise(self.resid_head(z_feats))  # (B, R)

        # Apply bounded activations for interpretability
        # times ≥0 via softplus; sustain level in (0,1) via sigmoid
        t_a, t_d, t_r, s_lvl = torch.split(raw_params, 1, dim=-1)
        param_vec = torch.cat([
            F.softplus(t_a),
            F.softplus(t_d),
            F.softplus(t_r),
            torch.sigmoid(s_lvl)
        ], dim=-1)

        z_a = torch.cat([param_vec, residual], dim=-1)   # (B, P+R)
        return z_a, param_vec


# --------------------------------------------------------------------------- #
# Differentiable ADSR kernel (parametric option)
# --------------------------------------------------------------------------- #

def adsr_kernel(params: torch.Tensor,
                env_sr: int = 1000,
                total_len: int = 1024) -> torch.Tensor:
    """
    Convert ADSR params to an envelope kernel suitable for convolution.

    Args
    ----
    params    : (B, 4) [t_a, t_d, t_r, s_lvl]
    env_sr    : sample-rate at which to build the envelope
    total_len : length of kernel in env-samples

    Returns
    -------
    env       : (B, 1, total_len) envelope kernel
    """
    B = params.size(0)
    t_a, t_d, t_r, s = params.t()          # each (B,)
    t_a  = (t_a * env_sr).long().clamp(1, total_len)
    t_d  = (t_d * env_sr).long().clamp(1, total_len)
    t_r  = (t_r * env_sr).long().clamp(1, total_len)

    env = torch.zeros(B, total_len, device=params.device)

    for b in range(B):
        A = t_a[b]
        D = t_d[b]
        R = t_r[b]
        S = s[b]

        # Attack
        env[b, :A] = torch.linspace(0, 1, A, device=params.device)
        # Decay
        end_d = A + D
        env[b, A:end_d] = torch.linspace(1, S, D, device=params.device)
        # Sustain (flat until start of release)
        start_r = total_len - R
        if start_r > end_d:
            env[b, end_d:start_r] = S
        # Release
        env[b, start_r:] = torch.linspace(S, 0, R, device=params.device)

    return env.unsqueeze(1)  # (B, 1, total_len)


# --------------------------------------------------------------------------- #
# FiLM Module (kept from original)
# --------------------------------------------------------------------------- #

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test original ADSREncoderV1
    print("\n=== Testing ADSREncoderV1 ===")
    B = 2
    wav = torch.randn(B, 1, 44100).to(device)             # dummy 1-second batch
    encoder_v1 = ADSREncoderV1()
    encoder_v1.to(device)
    z_a_v1 = encoder_v1(wav)
    print(f"ADSREncoderV1 output shape: {z_a_v1.shape}")  # → torch.Size([2, 64, 87])

    # Test new ADSREncoderV2
    print("\n=== Testing ADSREncoderV2 ===")
    wav = torch.randn(B, 1, 44100).to(device)  # dummy 1-second audio batch

    # Test without note mask (global pooling)
    encoder = ADSREncoderV2(hop=512, d_model=64, n_layers=3, param_dim=4, residual_dim=4)
    encoder.to(device)
    z_a, param_vec = encoder(wav)
    print(f"ADSREncoderV2 output shape: {z_a.shape}")  # → torch.Size([2, 8])
    print(f"Param vector shape: {param_vec.shape}")  # → torch.Size([2, 4])
    print(f"Param vector values: {param_vec[0]}")  # Show first batch params

    # Test with note mask
    N = 3  # number of notes
    T_frames = 87  # 44100 / 512 ≈ 87 frames
    note_mask = torch.randint(0, 2, (B, N, T_frames)).to(device).float()  # random binary mask
    z_a_masked, param_vec_masked = encoder(wav, note_mask)
    print(f"ADSREncoderV2 with note mask output shape: {z_a_masked.shape}")

    # Test ADSR kernel
    print("\n=== Testing ADSR kernel ===")
    params = torch.tensor([[0.1, 0.2, 0.3, 0.7],  # [t_a, t_d, t_r, s_lvl] for batch 1
                          [0.05, 0.15, 0.25, 0.8]], device=device)  # for batch 2
    env_kernel = adsr_kernel(params, env_sr=1000, total_len=1024)
    print(f"ADSR kernel shape: {env_kernel.shape}")  # → torch.Size([2, 1, 1024])
