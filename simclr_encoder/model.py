import os
import argparse
import torch
import torchvision
import librosa
import numpy as np
import torch.nn as nn
from pytorch_lightning import Trainer, LightningModule
from torchlars import LARS
from torchvision.models import vit_b_16

from loss import NT_Xent
from dataset import CLARTransform

    


class SimCLR(nn.Module):
    def __init__(
        self, 
        n_features=512, 
        projection_dim=128,
    ):
        super(SimCLR, self).__init__()

        self.encoder = vit_b_16(pretrained=True)
        self.encoder.conv_proj = nn.Conv2d(
            1,  # Single channel for mel-spectrograms
            self.encoder.conv_proj.out_channels,
            kernel_size=self.encoder.conv_proj.kernel_size,
            stride=self.encoder.conv_proj.stride,
            padding=self.encoder.conv_proj.padding,
            bias=False
        )

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.heads.head.in_features, n_features), #, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, projection_dim), #, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        
        return h_i, h_j, z_i, z_j
    


class ContrastiveLearning(LightningModule):
    def __init__(
        self, 
        args, 
        device
    ):
        super(ContrastiveLearning, self).__init__()

        self.args = args
        self.batch_size = args.batch_size
        self.save_dir = self.args.model_dict_save_dir
        self.my_device = torch.device(device)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Load Models
        self.transform = CLARTransform().to(device)
        self.model = SimCLR(
            n_features=self.args.encoder_output_dim,
            projection_dim=self.args.projection_dim, 
        ).to(device)
        
        self.criterion = NT_Xent(
            self.args.batch_size, 
            self.args.temperature, 
            world_size=1,
        ).to(device)
        
        self.save_hyperparameters()
        
        
    def get_spec_features(self, x, sr=16000):
        return np.float32(np.stack((
            librosa.power_to_db(np.abs(librosa.feature.melspectrogram(x, sr, hop_length=128))),
            librosa.amplitude_to_db(librosa.feature.mfcc(x, sr, n_mfcc=128, hop_length=128)),
            librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=254, hop_length=128)), ref=np.max)
        ), 0))
        

    def forward(self, x):
        x = x.to(self.my_device)
        x_i, x_j = self.transform(x)
        x_i, x_j = self.get_spec_features(x_i), self.get_spec_features(x_j)
        
        h_i, h_j, z_i, z_j = self.model(x_i, x_j) # SimCLR Model
        return h_i, h_j, z_i, z_j


    def training_step(self, batch, batch_idx):
        if batch.shape[0] != self.batch_size:
            return None
        
        _, _, z_i, z_j = self(batch)
        loss = self.criterion(z_i, z_j)
        
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size, sync_dist=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        if batch.shape[0] != self.batch_size:
            return None
        
        _, _, z_i, z_j = self(batch)
        loss = self.criterion(z_i, z_j)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size, sync_dist=True)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        if batch.shape[0] != self.batch_size:
            return None
        
        _, _, z_i, z_j = self(batch)
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