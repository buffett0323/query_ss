import os
import argparse
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import Trainer, LightningModule
from torchlars import LARS

from loss import NT_Xent
from dataset import SimCLRTransform

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
        self.block3 = GatedConvBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.block4 = GatedConvBlock(256, 512, kernel_size=3, stride=2, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(512, num_classes) # self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.block1(x))
        x = F.relu(self.block2(x))
        x = F.relu(self.block3(x))
        x = F.relu(self.block4(x))
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x



class DilatedCNN(nn.Module):
    def __init__(self, in_channels=2, encoder_output_dim=128):
        super(DilatedCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(256, encoder_output_dim, kernel_size=3, stride=1, padding=8, dilation=8)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(encoder_output_dim)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global pooling to get fixed-size embeddings
        # TODO: Reparameterize

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)  # Reduce spatial dimensions to 1x1
        return x.view(x.size(0), -1)  # Flatten to [batch_size, encoder_output_dim]
    


class SimCLR(nn.Module):
    def __init__(self, encoder, n_features, projection_dim):
        super(SimCLR, self).__init__()

        self.encoder = encoder

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, projection_dim, bias=False),
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
        self.save_dir = self.args.model_dict_save_dir
        self.my_device = torch.device(device)
        os.makedirs(self.save_dir, exist_ok=True)
        
        if args.encoder_name == "DilatedCNN":
            self.encoder = DilatedCNN(
                in_channels=self.args.channels,
                encoder_output_dim=self.args.encoder_output_dim,
            ).to(device)
            
        else: 
            self.encoder = GatedCNN(
                in_channels=self.args.channels, 
                num_classes=self.args.encoder_output_dim,
            ).to(device)
        
        self.model = SimCLR(
            encoder=self.encoder, 
            n_features=self.args.encoder_output_dim,
            projection_dim=self.args.projection_dim, 
        ).to(device)
        
        self.criterion = NT_Xent(
            self.args.batch_size, 
            self.args.temperature, 
            world_size=1,
        ).to(device)
        
        self.save_hyperparameters()
        self.batch_size = args.batch_size
    

    def forward(self, x_i, x_j):
        x_i, x_j = x_i.to(self.my_device), x_j.to(self.my_device)
        h_i, h_j, z_i, z_j = self.model(x_i, x_j) # SimCLR Model
        loss = self.criterion(z_i, z_j)
        return loss


    def training_step(self, batch, batch_idx):
        x_i, x_j = batch
        loss = self(x_i, x_j)
        self.log('train_loss', loss, on_step=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, stage='val')
    
    
    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, stage='test')

    
    def evaluate(self, batch, stage='val'):
        pass
    
    
    def configure_criterion(self):
        criterion = NT_Xent(self.args.batch_size, self.args.temperature)
        return criterion

    def configure_optimizers(self):
        scheduler = None
        if self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        
        elif self.args.optimizer == "LARS": # TODO
            # optimized using LARS with linear learning rate scaling
            # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
            learning_rate = 0.3 * self.args.batch_size / 256
            optimizer = LARS(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=self.args.weight_decay,
                exclude_from_weight_decay=["batch_normalization", "bias"],
            )

            # "decay the learning rate with the cosine decay schedule without restarts"
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.args.epochs, eta_min=0, last_epoch=-1
            )
        else:
            raise NotImplementedError

        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}
        
    def save_model(self, checkpoint_name="simclr.pth"):
        """Save model state dictionary."""
        save_path = os.path.join(self.save_dir, checkpoint_name)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model state saved to {save_path}")


    def load_model(self, checkpoint_name="simclr.pth"):
        """Load model state dictionary to continue training."""
        load_path = os.path.join(self.save_dir, checkpoint_name)
        if os.path.exists(load_path):
            self.model.load_state_dict(torch.load(load_path))
            print(f"Model state loaded from {load_path}")
        else:
            print(f"Checkpoint file not found at {load_path}")