import torch
import torch.nn as nn
import numpy as np
import yaml
import argparse
from transformer import DIT

class OneDEmbedder(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout=0.1):
        """
        Args:
            input_dim: Dimensionality of the input features (e.g., 64, 128, etc.)
            embed_dim: Desired embedding dimension for DiT input (e.g., 1536)
        """
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(input_dim, embed_dim, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.SiLU(),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, input_dim, T]
        Returns:
            Tensor of shape [B, T, embed_dim]
        """
        x = self.embed(x)          # [B, embed_dim, T]
        x = x.transpose(1, 2)      # [B, T, embed_dim]
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for dit training')
    parser.add_argument('--config', dest='config_path',
                        default='dit/config.yaml', type=str)
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    dit_model_config = config['dit_params']
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    ########################
    model = DIT(
        image_height=64,
        image_width=80,
        im_channels=1,
        config=dit_model_config
    ).to(device)
    oned_embedder = OneDEmbedder(
        32, 32
    ).to(device)


    x = torch.randn(4, 1, 64, 80).to(device)
    t = torch.randint(0, 1000, (4,)).to(device)
    out = model(x, t, None)
    print(out.shape)
    print(model)
    # # 0. Time Embedding
    # t = torch.randint(0, 1000, (4,)).to(device)

    # # Conditions
    # spk = torch.randn(4, 2048).to(device)
    # # bp_notes = torch.randn(4, 88, 251).to(device)

    # # 1. 8-layer WN Output
    # enc_output = torch.randn(4, 80, 200).to(device)

    # # 2. Masked Input Spectrogram
    # masked_prompt = torch.randn(4, 80, 200).to(device)

    # # 3. Noisy latent
    # z = torch.randn(4, 80, 200).to(device) # B, D, T

    # stacked_input = torch.cat([enc_output, masked_prompt, z], dim=2)

    # # 4. OneDEmbedder
    # stacked_input = oned_embedder(stacked_input).unsqueeze(1)
    # print("DiT Input shape:", stacked_input.shape) # 4, 600, 80

    # # 5. DiT
    # out = model(stacked_input, t, spk)
    # print(out.shape) # 4, 80, 600
