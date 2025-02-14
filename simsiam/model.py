import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch_models import Wavegram_Logmel128_Cnn14
from swin_transformer import SwinTransformer
from utils import *


class SimSiam(nn.Module):
    """
    SimSiam model with Wavegram_Logmel128_Cnn14 as the encoder.
    """
    def __init__(self, args, dim=2048, pred_dim=512):
        super(SimSiam, self).__init__()
        self.args = args

        # ** Encoders **
        if self.args.encoder_name == "Wavegram_Logmel128_Cnn14":
            self.encoder = Wavegram_Logmel128_Cnn14(
                sample_rate=self.args.sample_rate,
                window_size=self.args.window_size,
                hop_size=self.args.hop_length,
                mel_bins=self.args.n_mels,
                fmin=self.args.fmin,
                fmax=self.args.fmax,
                classes_num=dim  # Output embedding dimension
            )
            # Remove the classification head to use raw feature embeddings
            prev_dim = self.encoder.fc1.weight.shape[1] # Extracting feature dimension # 2048
            self.encoder.fc1 = nn.Identity() 
        
        elif self.args.encoder_name == "SwinTransformer":
            self.encoder = SwinTransformer(img_size=224, window_size=7, in_chans=1, num_classes=0)
            prev_dim = self.encoder.num_features
            # num_classes = 0 --> self.head = nn.Identity
            
        else:
            self.encoder = nn.Identity()
        
        print("prev_dim:", prev_dim) 
        
                
        # **Build a separate 3-layer projector**
        self.projector = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # First layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # Second layer
            nn.Linear(prev_dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False)  # Output layer (no affine)
        )

        # **Build a 2-layer predictor**
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # Hidden layer
            nn.Linear(pred_dim, dim)  # Output layer
        )

    def forward(self, x1, x2):
        """
        Forward pass for SimSiam model.
        """
        # transform_rs = transforms.Resize((256, 256))
        # x1 = transform_rs(x1).unsqueeze(1)
        # x2 = transform_rs(x2).unsqueeze(1)
        
        # Compute features for both views
        z1 = self.projector(self.encoder(x1))  # NxC
        z2 = self.projector(self.encoder(x2))  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1.detach(), z2.detach()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="SimCLR Encoder")

    config = yaml_config_hook("config/ssbp_6secs.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    
    # Load models
    model = SimSiam(args)#.to(device)

    
    x = torch.randn([16, 1, 256, 256])#.to(device)
    res = model(x, x)
    print(res[0].shape, res[2].shape)