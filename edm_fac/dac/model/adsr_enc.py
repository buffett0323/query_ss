import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class ADSREncoder(nn.Module):
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
    B = 2
    wav = torch.randn(B, 1, 44100).to(device)             # dummy 1-second batch
    encoder = ADSREncoder()
    encoder.to(device)
    z_a = encoder(wav)
    print(z_a.shape)                           # → torch.Size([2, 64, 87])
