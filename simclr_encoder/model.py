import os
import argparse
import torch
import torchvision
import librosa
import numpy as np
import torch.nn as nn
from pytorch_lightning import Trainer, LightningModule
import torch.nn.functional as F
import torchaudio.transforms as T
import torchvision.transforms as transforms
from torchlars import LARS
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models.vision_transformer import VisionTransformer

from loss import NT_Xent
from dataset import CLARTransform
from torch_models import Wavegram_Logmel128_Cnn14

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
        # TODO: Reparameterize

    def forward(self, x):
        x = F.relu(self.block1(x))
        x = F.relu(self.block2(x))
        x = F.relu(self.block3(x))
        x = F.relu(self.block4(x))
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x



    

class SimCLR(nn.Module):
    def __init__(
        self,
        args,
        n_features=512, 
        projection_dim=128,
    ):
        super(SimCLR, self).__init__()
        self.args = args
        if self.args.encoder_name == "GatedCNN":
            self.encoder = GatedCNN(
                in_channels=self.args.channels, 
                num_classes=self.args.encoder_output_dim,
            )
        else: # Wavegram_Logmel128_Cnn14
            self.encoder = Wavegram_Logmel128_Cnn14(
                sample_rate=self.args.sample_rate, 
                window_size=self.args.window_size, 
                hop_size=self.args.hop_length, 
                mel_bins=self.args.n_mels, 
                fmin=self.args.fmin,
                fmax=self.args.fmax,
                classes_num=self.args.encoder_output_dim,
            )
            
        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(n_features, n_features), #, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, projection_dim), #, bias=False),
        )

    def forward(self, x_i, x_j):
        print("xixj", x_i.shape, x_j.shape)

        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        print("hihj", h_i.shape, h_j.shape)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        
        return h_i, h_j, z_i, z_j
    


class ContrastiveLearning(LightningModule):
    def __init__(
        self, 
        args, 
        device,
    ):
        super(ContrastiveLearning, self).__init__()

        self.args = args
        self.batch_size = args.batch_size
        self.save_dir = self.args.model_dict_save_dir
        self.my_device = torch.device(device)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Load Models
        self.model = SimCLR(
            args=args,
            n_features=self.args.encoder_output_dim,
            projection_dim=self.args.projection_dim, 
        ).to(device)
        
        self.criterion = NT_Xent(
            self.args.batch_size, 
            self.args.temperature, 
            world_size=1,
        ).to(device)
        
        # self.save_hyperparameters()
        
        self.mel_transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=self.args.sample_rate,
                n_fft=self.args.n_fft,
                hop_length=self.args.hop_length,
                n_mels=self.args.n_mels,
                power=2.0
            ),
            T.AmplitudeToDB()
        ).to(device)
            

    def forward(self, x_i, x_j):
        if x_i.dtype != torch.float32:
            x_i = x_i.to(torch.float32)
        if x_j.dtype != torch.float32:
            x_j = x_j.to(torch.float32)
        x_i, x_j = self.mel_transform(x_i).unsqueeze(1), self.mel_transform(x_j).unsqueeze(1)
        h_i, h_j, z_i, z_j = self.model(x_i, x_j) # SimCLR Model
        return h_i, h_j, z_i, z_j


    def training_step(self, batch, batch_idx):
        x_i, x_j = batch
        if x_i.shape[0] != self.batch_size:
            return None
        
        _, _, z_i, z_j = self(x_i, x_j)
        loss = self.criterion(z_i, z_j)
        
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size, sync_dist=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x_i, x_j = batch
        if x_i.shape[0] != self.batch_size:
            return None
        
        _, _, z_i, z_j = self(x_i, x_j)
        loss = self.criterion(z_i, z_j)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size, sync_dist=True)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        x_i, x_j = batch
        if x_i.shape[0] != self.batch_size:
            return None
        
        _, _, z_i, z_j = self(x_i, x_j)
        loss = self.criterion(z_i, z_j)

        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size, sync_dist=True)
        return loss

    
    def configure_criterion(self):
        criterion = NT_Xent(self.args.batch_size, self.args.temperature)
        return criterion


    def configure_optimizers(self):
        scheduler = None
        if self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        elif self.args.optimizer == "LARS": # TODO
            # optimized using LARS with linear learning rate scaling
            # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
            learning_rate = 0.3 * self.args.batch_size / 256
            
            # Base optimizer: SGD with momentum
            base_optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=self.args.weight_decay,
            )


            # Wrap the base optimizer with LARS
            optimizer = LARS(
                optimizer=base_optimizer,
                eps=1e-8,  # Epsilon for numerical stability
                trust_coef=0.001,  # Trust coefficient
            )

            # Cosine annealing scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                base_optimizer, T_max=self.args.epochs, eta_min=0, last_epoch=-1
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