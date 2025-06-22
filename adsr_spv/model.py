import math
from typing import Literal, Tuple

import torch
import torch.nn as nn
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(
        self, size: int, num_pos: int, init: Literal["zeros", "norm0.02"] = "zeros"
    ):
        super().__init__()

        if init == "zeros":
            pe = torch.zeros(1, num_pos, size)
        else:
            pe = torch.randn(1, num_pos, size) * 0.02

        self.pe = nn.Parameter(pe)

    def penalty(self) -> torch.Tensor:
        # structured sparsity
        return self.pe.norm(2.0, dim=-1).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.pe[:, : x.shape[1], :]
        return x + pe


class KSinParamToTokenProjection(nn.Module):
    def __init__(self, d_model: int, params_per_token: int = 2):
        super().__init__()
        self.forward_proj = nn.Linear(params_per_token, d_model)
        self.backward_proj = nn.Linear(d_model, params_per_token)
        self.params_per_token = params_per_token

    def param_to_token(self, x: torch.Tensor) -> torch.Tensor:
        k = x.shape[-1] // self.params_per_token
        x = rearrange(x, "b (d k) -> b k d", k=k)

        x = self.forward_proj(x)

        return x

    def token_to_param(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backward_proj(x)
        x = rearrange(x, "b k d -> b (d k)", d=self.params_per_token)
        return x

    def penalty(self) -> torch.Tensor:
        return 0.0


class LearntProjection(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_token: int,
        num_params: int,
        num_tokens: int,
        initial_ffn: bool = True,
        final_ffn: bool = True,
    ):
        super().__init__()

        assignment = torch.full(
            (num_tokens, num_params), 1.0 / math.sqrt(num_tokens * num_params)
        )
        assignment = assignment + 1e-4 * torch.randn_like(assignment)
        self._assignment = nn.Parameter(assignment)

        proj = torch.randn(1, d_token) / math.sqrt(d_token)
        proj = proj.repeat(num_params, 1)
        proj = proj + 1e-4 * torch.randn_like(proj)

        self._in_projection = nn.Parameter(proj.clone())
        self._out_projection = nn.Parameter(proj.T.clone())


        if initial_ffn:
            self.initial_ffn = nn.Sequential(
                nn.Linear(d_token, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.initial_ffn = None

        if final_ffn:
            self.final_ffn = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_token),
            )
        elif d_token == d_model:
            self.final_ffn = None
        else:
            self.final_ffn = nn.Linear(d_model, d_token)

    @property
    def assignment(self):
        return self._assignment

    @property
    def in_projection(self):
        return self._in_projection

    @property
    def out_projection(self):
        return self._out_projection

    def param_to_token(self, x: torch.Tensor) -> torch.Tensor:
        values = torch.einsum("bn,nd->bnd", x, self.in_projection)

        if self.initial_ffn is not None:
            values = self.initial_ffn(values)

        tokens = torch.einsum("bnd,kn->bkd", values, self.assignment)

        return tokens

    def token_to_param(self, x: torch.Tensor) -> torch.Tensor:
        deassigned = torch.einsum("bkd,kn->bnd", x, self.assignment)

        if self.final_ffn is not None:
            deassigned = self.final_ffn(deassigned)

        return torch.einsum("bnd,dn->bn", deassigned, self.out_projection)

    def penalty(self) -> torch.Tensor:
        # we apply L1 penalty to the assignment matrix
        penalty = self.assignment.abs().mean()

        return penalty





class AdaptiveLayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, conditioning_dim: int, *args, **kwargs):
        super().__init__(dim, *args, **kwargs)
        self.shift = nn.Linear(conditioning_dim, dim)
        self.scale = nn.Linear(conditioning_dim, dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        shift = self.shift(z)
        scale = self.scale(z)
        x = super().forward(x)
        return x * scale + shift


class DiTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        conditioning_dim: int,
        num_heads: int,
        d_ff: int,
        norm: Literal["layer", "rms"] = "layer",
        first_norm: bool = True,
        adaln_mode: Literal["basic", "zero", "res"] = "basic",
        zero_init: bool = True,
    ):
        super().__init__()
        if first_norm:
            self.norm1 = (
                nn.LayerNorm(d_model) if norm == "layer" else nn.RMSNorm(d_model)
            )
        else:
            self.norm1 = nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if norm == "layer" else nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        # self.attn = MultiheadAttention(d_model, num_heads)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        cond_out_dim = d_model * 6 if adaln_mode != "res" else d_model * 4
        self.adaln_mode = adaln_mode
        self.cond = nn.Sequential(
            nn.GELU(),
            nn.Linear(conditioning_dim, cond_out_dim),
        )

        self._init_adaln(adaln_mode)
        self._init_ffn(zero_init)
        self._init_attn(zero_init)

    def _init_adaln(self, mode: Literal["basic", "zero"]):
        if mode == "zero":
            nn.init.constant_(self.cond[-1].weight, 0.0)
            nn.init.constant_(self.cond[-1].bias, 0.0)

    def _init_ffn(self, zero_init: bool):
        nn.init.xavier_normal_(self.ff[0].weight)
        nn.init.zeros_(self.ff[0].bias)
        nn.init.zeros_(self.ff[-1].bias)

        if zero_init:
            nn.init.zeros_(self.ff[-1].weight)
        else:
            nn.init.xavier_normal_(self.ff[-1].weight)

    def _init_attn(self, zero_init: bool):
        if zero_init:
            nn.init.zeros_(self.attn.out_proj.weight)
            nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.adaln_mode == "res":
            g1, b1, g2, b2 = self.cond(z)[:, None].chunk(4, dim=-1)
        else:
            g1, b1, a1, g2, b2, a2 = self.cond(z)[:, None].chunk(6, dim=-1)

        res = x
        x = self.norm1(x)
        x = g1 * x + b1
        x = self.attn(x, x, x)[0]

        if self.adaln_mode == "res":
            x = x + res
        else:
            x = a1 * x + res

        res = x
        x = self.norm2(x)
        x = g2 * x + b2
        x = self.ff(x)

        if self.adaln_mode == "res":
            x = x + res
        else:
            x = a2 * x + res

        return x



class SinusoidalEncoding(nn.Module):
    """A sinusoidal encoding of scalar values centered around zero."""

    def __init__(self, d_model: int):
        super().__init__()

        half = d_model // 2
        k = torch.arange(0, half)
        basis = 1 / torch.pow(10000, k / half)

        self.register_buffer("basis", basis[None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == 1:
            basis = self.basis
        else:
            basis = self.basis[None, :]
            x = x[:, :, None]

        cos_part = torch.cos(x * self.basis)
        sin_part = torch.sin(x * self.basis)
        return torch.cat([cos_part, sin_part], dim=-1)


class ConcatConditioning(nn.Module):
    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.cat([z, t], dim=-1)


class SinusoidalConditioning(nn.Module):
    def __init__(self, d_model: int, d_enc: int):
        super().__init__()
        self.d_model = d_model
        self.sin = SinusoidalEncoding(d_enc)
        self.mlp = nn.Sequential(
            nn.Linear(d_enc, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = self.sin(t)
        t = self.mlp(t)
        return z + t


class MutualAttentionProjection(nn.Module):
    """
    p (b, n) parameters
    sinusoidal embed -> MLP
    +
    pos embed

    to get (b, n, d) tokens

    then concat with k learnt tokens for (b, n + k, d)
    apply self attn and take last k for (b, k, d)

    pass through transformer

    at output concat with n learnt tokens for (b, n + k, d)
    apply self attn and take first n
    final ffn to 1d
    """

    def __init__(self, d_model: int, num_params: int, num_tokens: int):
        super().__init__()

        scale = 1 / math.sqrt(d_model)
        self.token_queries = nn.Parameter(torch.randn(1, num_tokens, d_model) * scale)
        self.param_queries = nn.Parameter(torch.randn(1, num_params, d_model) * scale)

        self.in_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.out_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)

        self.sin = SinusoidalConditioning(d_model, 256)
        self.pos = nn.Parameter(torch.randn(1, num_params, d_model) * scale)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def param_to_token(self, params: torch.Tensor) -> torch.Tensor:
        param_encodings = self.sin(self.pos, params)
        token_queries = self.token_queries.repeat(params.shape[0], 1, 1)
        in_seq = torch.cat([param_encodings, token_queries], dim=1)
        token_encodings, _ = self.in_attn(in_seq, param_encodings, param_encodings)
        return token_encodings[:, -self.token_queries.shape[1] :]

    def token_to_param(self, tokens: torch.Tensor) -> torch.Tensor:
        param_queries = self.param_queries.repeat(tokens.shape[0], 1, 1)
        in_seq = torch.cat([param_queries, tokens], dim=1)
        param_encodings, _ = self.out_attn(in_seq, tokens, tokens)
        param_encodings = param_encodings[:, : self.param_queries.shape[1]]
        params = self.mlp(param_encodings).squeeze(-1)
        return params

    def penalty(self):
        return 0.0



class PatchEmbed(nn.Module):
    """Convolutional patch encoder like in ViT, with overlap from AST.
    Difference is we zero pad up to next whole patch.
    """

    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        d_model: int,
        spec_shape: Tuple[int] = (128, 401),
    ):
        super().__init__()
        assert stride < patch_size, "Overlap must be less than patch size"

        self.patch_size = patch_size

        mel_padding = (stride - (spec_shape[0] - patch_size)) % stride
        time_padding = (stride - (spec_shape[1] - patch_size)) % stride

        self.pad = nn.ZeroPad2d((0, mel_padding, 0, time_padding))
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=stride,
        )

        self.num_tokens = self._get_num_tokens(in_channels, spec_shape)

    def _get_num_tokens(self, in_channels, spec_shape):
        x = torch.randn(
            1, in_channels, *spec_shape, device=self.projection.weight.device
        )
        out_shape = self.projection(self.pad(x)).shape
        return math.prod(out_shape[-2:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class AudioSpectrogramTransformer(nn.Module):
    """Based on the AST from https://arxiv.org/abs/2104.01778, but adapted to pre-norm
    transformer.

    Components:
        1. patch split with overlap
        2. linear token projection
        3. class (embedding) token
        4. transformer encoder
        5. output linear projection
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 16,
        n_conditioning_outputs: int = 12,
        patch_size: int = 16,
        patch_stride: int = 10,
        input_channels: int = 2,
        spec_shape: Tuple[int] = (128, 401),
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            stride=patch_stride,
            in_channels=input_channels,
            d_model=d_model,
            spec_shape=spec_shape,
        )

        self.positional_encoding = PositionalEncoding(
            d_model,
            self.patch_embed.num_tokens + n_conditioning_outputs,
            init="norm0.02",
        )
        self.embed_tokens = nn.Parameter(
            torch.empty(1, n_conditioning_outputs, d_model).normal_(0.0, 1e-6)
        )

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model,
                    n_heads,
                    d_model,
                    0.0,
                    "gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # produce input sequence
        x = self.patch_embed(x)

        embed_tokens = self.embed_tokens.expand(x.shape[0], -1, -1)
        x = torch.cat((embed_tokens, x), dim=1)

        x = self.positional_encoding(x)

        # apply transformer
        for block in self.blocks:
            x = block(x)

        # take just the embed tokens
        x = x[:, : self.embed_tokens.shape[1]]
        x = self.out_proj(x)

        return x


class ASTWithProjectionHead(AudioSpectrogramTransformer):
    """Based on the AST from https://arxiv.org/abs/2104.01778, but adapted to pre-norm
    transformer.

    Components:
        1. patch split with overlap
        2. linear token projection
        3. class (embedding) token
        4. transformer encoder
        5. output linear projection
    """

    def __init__(
        self,
        d_model: int = 768,
        d_out: int = 16,
        n_heads: int = 8,
        n_layers: int = 16,
        patch_size: int = 16,
        patch_stride: int = 10,
        input_channels: int = 2,
        spec_shape: Tuple[int] = (128, 401),
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_conditioning_outputs=1,
            patch_size=patch_size,
            patch_stride=patch_stride,
            input_channels=input_channels,
            spec_shape=spec_shape,
        )

        self.prediction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.final_proj = nn.Linear(d_model, d_out)

        # enforce sustain ∈ [0,1] via sigmoid; others positive via softplus.
        self.softplus = nn.Softplus()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)[:, 0]

        x = self.prediction_head(x) + x
        out = self.final_proj(x)

        sustain_raw = out[:, 2:3]
        sustain = torch.sigmoid(sustain_raw)  # 0‑1 level
        times_raw = torch.relu(out[:, [0, 1, 3]])  # ensure non‑neg
        times = self.softplus(times_raw)          # smooth >0
        return torch.cat([times[:, 0:1], times[:, 1:2], sustain, times[:, 2:3]], dim=1)



if __name__ == "__main__":
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    model = ASTWithProjectionHead(
        d_model=1024,
        d_out=4, # A,D,S,R
        n_heads=8,
        n_layers=16,
        patch_size=16,
        patch_stride=10,
        input_channels=1,
    )
    model.to(device)
    model.eval()

    x = torch.randn(8, 1, 128, 256).to(device)
    x = model(x)
    print(x.shape)
    print(x)
