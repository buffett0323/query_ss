import os
import argparse
import torch
import torchvision
import librosa
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio.transforms as T
import torchvision.transforms as transforms
# from torchlars import LARS #from torch.optim import LARS
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models.vision_transformer import VisionTransformer
from torch.optim import SGD, Adam
from pytorch_lightning import Trainer, LightningModule
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from loss import NT_Xent, ContrastiveLoss
from utils import define_param_groups
from dataset import CLARTransform
from torch_models import Wavegram_Logmel_Cnn14, Wavegram_Logmel128_Cnn14




class SimCLR(nn.Module):
    def __init__(
        self,
        args,
        n_features=512,
        projection_dim=128,
    ):
        super(SimCLR, self).__init__()
        self.args = args
        if self.args.encoder_name == "Wavegram_Logmel128_Cnn14": # Wavegram_Logmel128_Cnn14
            self.encoder = Wavegram_Logmel128_Cnn14(
                sample_rate=self.args.sample_rate,
                window_size=self.args.window_size,
                hop_size=self.args.hop_length,
                mel_bins=128, #self.args.n_mels,
                fmin=self.args.fmin,
                fmax=self.args.fmax,
                classes_num=n_features,
            )
        elif self.args.encoder_name == "Wavegram_Logmel_Cnn14": # Wavegram_Logmel128_Cnn14
            self.encoder = Wavegram_Logmel_Cnn14(
                sample_rate=self.args.sample_rate,
                window_size=self.args.window_size,
                hop_size=self.args.hop_length,
                mel_bins=64, #self.args.n_mels,
                fmin=self.args.fmin,
                fmax=self.args.fmax,
                classes_num=n_features,
            )
        else:
            self.encoder = None

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(n_features, n_features), #, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, projection_dim), #, bias=False),
        )


        # add mlp projection head
        self.projection = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=n_features),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
            nn.Linear(in_features=n_features, out_features=projection_dim),
            nn.BatchNorm1d(projection_dim),
        )


    def forward(self, x, return_embedding=False):
        embedding = self.encoder(x)

        if return_embedding:
            return embedding

        return self.projection(embedding) # return self.projector(embedding)



class SimCLR_pl(LightningModule):
    def __init__(
        self,
        args,
        device,
    ):
        super(SimCLR_pl, self).__init__()

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
        )

        # self.criterion = NT_Xent(args.batch_size, args.temperature, world_size=1,)
        self.criterion = ContrastiveLoss(args.batch_size, temperature=args.temperature)

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
        )


    def forward(self, x, return_embedding=False):
        if self.args.need_transform:
            x = self.mel_transform(x).unsqueeze(1)
        return self.model(x, return_embedding) # SimCLR Model


    def training_step(self, batch, batch_idx):
        x_i, x_j = batch
        if x_i.shape[0] != self.batch_size:
            return None

        z_i, z_j = self(x_i), self(x_j)
        loss = self.criterion(z_i, z_j)

        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x_i, x_j = batch
        if x_i.shape[0] != self.batch_size:
            return None

        z_i, z_j = self(x_i), self(x_j)
        loss = self.criterion(z_i, z_j)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size, sync_dist=True)
        return loss


    def test_step(self, batch, batch_idx):
        x_i, x_j = batch
        if x_i.shape[0] != self.batch_size:
            return None

        z_i, z_j = self(x_i), self(x_j)
        loss = self.criterion(z_i, z_j)

        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size, sync_dist=True)
        return loss



    def configure_optimizers(self):
        max_epochs = int(self.args.max_epochs)
        param_groups = define_param_groups(self.model, self.args.weight_decay, 'adam')
        lr = self.args.lr
        optimizer = Adam(param_groups, lr=lr, weight_decay=self.args.weight_decay)

        print(f'Optimizer Adam, '
                f'Learning Rate {lr}, '
                f'Effective batch size {self.args.batch_size * self.args.gradient_accumulation_steps}')

        scheduler_warmup = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.args.warmup_epochs,
            max_epochs=max_epochs,
            warmup_start_lr=0.0
            )

        return [optimizer], [scheduler_warmup]


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



    @classmethod
    def from_config(cls, checkpoint_path, args, device):
        model = cls(args, device)
        model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
        return model
