import os
import argparse

from pathlib import Path
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from byol_pytorch import BYOL
import pytorch_lightning as pl
from util import N_MELS, BATCH_SIZE, NUM_WORKERS, NUM_GPUS, EPOCHS, LR
from dataset import ADSRDataset




# Custom BYOL model for audio
class AudioBYOL(nn.Module):
    def __init__(self, encoder, **kwargs):
        super().__init__()
        self.learner = BYOL(
            encoder,
            image_size=N_MELS,  # Using n_mels as the height dimension
            hidden_layer='avgpool',
            projection_size=256,
            projection_hidden_size=4096,
            moving_average_decay=0.99,
            **kwargs
        )

    def forward(self, x):
        return self.learner(x)
    

# PyTorch Lightning module
class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = AudioBYOL(net, **kwargs)

    def forward(self, x):
        return self.learner(x)

    def training_step(self, batch, _):
        mel1, mel2 = batch
        loss = self.forward(mel1) + self.forward(mel2)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.learner.use_momentum:
            self.learner.learner.update_moving_average()

# Main training script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BYOL Audio ADSR Training')
    parser.add_argument('--audio_folder', type=str, required=True,
                       help='path to your folder of audio files for self-supervised learning')
    args = parser.parse_args()

    # Create dataset and dataloader
    ds = ADSRDataset(args.audio_folder)
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    # Create encoder model (you should replace this with your ADSR encoder)
    encoder = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )

    # Create and train model
    model = SelfSupervisedLearner(
        encoder,
        image_size=N_MELS,
        hidden_layer='avgpool',
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99
    )

    trainer = pl.Trainer(
        gpus=NUM_GPUS,
        max_epochs=EPOCHS,
        accumulate_grad_batches=1,
        sync_batchnorm=True
    )

    trainer.fit(model, train_loader)
