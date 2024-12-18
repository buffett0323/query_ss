import torch
import torch.nn as nn
import torch.nn.functional as F

# Adaptive LayerNorm for conditioning
class AdaptiveLayerNorm(nn.Module):
    def __init__(self, normalized_shape, condition_dim):
        super(AdaptiveLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.condition_proj = nn.Linear(condition_dim, 2 * normalized_shape)  # Scale and Shift

    def forward(self, x, condition):
        # condition: [batch_size, condition_dim]
        scale_shift = self.condition_proj(condition)  # [batch_size, 2 * normalized_shape]
        scale, shift = scale_shift.chunk(2, dim=-1)   # Split into scale and shift factors

        x = self.layer_norm(x)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# Transformer Block with Adaptive LayerNorm
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, condition_dim, ff_dim=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.adaLN1 = AdaptiveLayerNorm(dim, condition_dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim)
        )
        self.adaLN2 = AdaptiveLayerNorm(dim, condition_dim)

    def forward(self, x, condition):
        # Self-attention with conditioning
        x_residual = x
        x = self.adaLN1(x, condition)
        x = self.self_attn(x, x, x, need_weights=False)[0] + x_residual  # Skip connection

        # Feedforward network with conditioning
        x_residual = x
        x = self.adaLN2(x, condition)
        x = self.ffn(x) + x_residual  # Skip connection

        return x

# DiT Architecture
class DiT(nn.Module):
    def __init__(self, dim, num_heads, condition_dim, num_blocks=3, ff_dim=2048, dropout=0.1):
        super(DiT, self).__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, condition_dim, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.time_embed = nn.Sequential(
            nn.Linear(1, condition_dim),
            nn.ReLU(),
            nn.Linear(condition_dim, condition_dim)
        )

    def forward(self, zm_t, sc, t):
        """
        Args:
            zm_t: Tensor of shape [batch_size, seq_len, dim] (input feature sequence)
            sc: Conditioning vector of shape [batch_size, condition_dim]
            t: Diffusion step scalar [batch_size, 1]
        Returns:
            zm_t: Transformed tensor of shape [batch_size, seq_len, dim]
        """
        # Embed the diffusion step t and combine with sc
        t_embed = self.time_embed(t)  # Shape: [batch_size, condition_dim]
        print(t_embed.shape)
        condition = sc + t_embed      # Combine the step and source-level condition

        # Pass through transformer blocks
        for block in self.blocks:
            zm_t = block(zm_t, condition)

        return zm_t


# Define input dimensions
batch_size = 4
seq_len = 25
dim = 512
condition_dim = 256
num_heads = 4

# Create inputs
zm_t = torch.randn(batch_size, seq_len, dim)  # Input sequence
sc = torch.randn(batch_size, condition_dim)  # Conditioning vector
t = torch.randint(0, 1000, (batch_size, 1)).float()  # Diffusion step

# Initialize DiT
dit = DiT(dim=dim, num_heads=num_heads, condition_dim=condition_dim, num_blocks=3)

# Forward pass
output = dit(zm_t, sc, t)
print("Output shape:", output.shape)  # Expected: [batch_size, seq_len, dim]
