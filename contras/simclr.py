import argparse
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
# from simclr.modules.resnet_hacks import modify_resnet_model
# from simclr.modules.identity import Identity

# SimCLR
from loss import NT_Xent

# from simclr.modules import NT_Xent, get_resnet
# from simclr.modules.transformations import TransformsSimCLR
# from simclr.modules.sync_batchnorm import convert_model

class GatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        A single block of Gated Convolutional Neural Network.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding for the convolution.
        """
        super(GatedConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gate = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        conv_out = self.conv(x)
        gate_out = torch.sigmoid(self.gate(x))
        return conv_out * gate_out


class GatedCNN(nn.Module):
    def __init__(self, in_channels=2, num_classes=128):
        """
        Gated Convolutional Neural Network for feature extraction.

        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of output features.
        """
        super(GatedCNN, self).__init__()
        self.block1 = GatedConvBlock(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.block2 = GatedConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        # self.block3 = GatedConvBlock(128, 256, kernel_size=3, stride=2, padding=1)
        # self.block4 = GatedConvBlock(256, 512, kernel_size=3, stride=2, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(128, num_classes) # self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.block1(x))
        x = F.relu(self.block2(x))
        # x = F.relu(self.block3(x))
        # x = F.relu(self.block4(x))
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x



class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, n_features, projection_dim):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        
        return h_i, h_j, z_i, z_j
    


class ContrastiveLearning(LightningModule):
    def __init__(self, args, device):
        super().__init__()

        self.args = args
        
        self.encoder = GatedCNN(
            in_channels=self.args.channels, 
            num_classes=self.args.n_features,
        ).to(device)
        
        self.model = SimCLR(
            self.encoder, 
            self.args.n_features,
            self.args.projection_dim, 
        ).to(device)
        
        self.criterion = NT_Xent(
            self.args.batch_size, self.args.temperature, world_size=1
        ).to(device)

    def forward(self, x_i, x_j):
        h_i, h_j, z_i, z_j = self.model(x_i, x_j)
        loss = self.criterion(z_i, z_j)
        print("Loss:", loss.item())
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x_i, x_j = batch
        print(x_i.shape, x_j.shape)
        loss = self(x_i, x_j)
        return loss

    def configure_criterion(self):
        criterion = NT_Xent(self.args.batch_size, self.args.temperature)
        return criterion

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        return {"optimizer": optimizer}