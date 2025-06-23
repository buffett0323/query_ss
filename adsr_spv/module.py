import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import nnAudio.features
from typing import Dict, Any
import wandb

from model import ASTWithProjectionHead
from norm import NormalizeBatch, PrecomputedNorm

class EfficientADSRLoss(nn.Module):
    """Ultra-efficient ADSR loss computation."""
    
    def __init__(self):
        super().__init__()
        # Pre-allocate tensors for maximum efficiency
        self.register_buffer('param_names', torch.tensor([0, 1, 2, 3]))  # attack, decay, sustain, release
        
    def forward(self, pred, target):
        """
        Compute total and individual ADSR losses in one pass.
        
        Args:
            pred: [batch_size, 4] predicted ADSR values
            target: [batch_size, 4] target ADSR values
            
        Returns:
            total_loss: scalar tensor
            individual_losses: [4] tensor with losses for each parameter
        """
        # Single computation of squared differences
        diff_squared = (pred - target) ** 2
        
        # Total loss (mean across all dimensions)
        total_loss = diff_squared.mean()
        
        # Individual losses (mean across batch dimension)
        individual_losses = diff_squared.mean(dim=0)  # [4]
        
        return total_loss, individual_losses


class ADSRLightningModule(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Initialize model
        self.model = ASTWithProjectionHead(
            d_model=config['d_model'],
            d_out=config['d_out'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            patch_size=config['patch_size'],
            patch_stride=config['patch_stride'],
            input_channels=config['input_channels'],
            spec_shape=tuple(config['spec_shape'])
        )
        
        # Initialize mel spectrogram converter
        self.to_spec = nnAudio.features.MelSpectrogram(
            sr=config['sr'], #44100,
            n_mels=config['n_mels'], #128,
            fmin=config['fmin'],
            fmax=config['fmax'],
            hop_length=config['hop_length'],
            n_fft=config['n_fft'],
            window='hann',
            center=True,
            power=2.0,
        ).to(self.device)
        
        # Initialize efficient loss function
        self.loss_fn = EfficientADSRLoss()
        self.pre_norm = PrecomputedNorm(np.array([config['mel_mean'], config['mel_std']]))
        self.post_norm = NormalizeBatch()
        
        # ADSR parameter names for logging
        self.adsr_params = ["attack", "decay", "sustain", "release"]
        
        # Pre-allocate logging dictionaries to avoid repeated dict creation
        self._train_log_dict = {}
        self._val_log_dict = {}
        
        
    def wav2mel(self, wav):
        with torch.no_grad():
            mel = self.to_spec(wav).unsqueeze(1) # [BS, 1, 128, 256]
        eps = torch.finfo(mel.dtype).eps
        mel = (mel + eps).log()
        mel = self.pre_norm(mel)
        mel = self.post_norm(mel)
        
        if mel.isnan().any():
            print("mel being nan detected")
            return None
        return mel
        
    
    def forward(self, mel):
        return self.model(mel)
    
    
    
    def _prepare_log_dict(self, prefix, total_loss, individual_losses):
        """Pre-allocate and fill logging dictionary efficiently."""
        log_dict = getattr(self, f'_{prefix}_log_dict')
        log_dict.clear()  # Reuse the same dict
        
        log_dict[f'{prefix}_loss'] = total_loss
        for i, param_name in enumerate(self.adsr_params):
            log_dict[f'{prefix}_{param_name}_loss'] = individual_losses[i]
        
        return log_dict
    
    
    
    def training_step(self, batch, batch_idx):
        wav, adsr_gt = batch
        
        # Ensure wav is on the correct device
        wav = wav.to(self.device)
        adsr_gt = adsr_gt.to(self.device)
        
        mel = self.wav2mel(wav)

        # Forward pass
        adsr_pred = self.forward(mel)
        
        # Ultra-efficient loss calculation
        total_loss, individual_losses = self.loss_fn(adsr_pred, adsr_gt)
        
        # Efficient logging
        log_dict = self._prepare_log_dict('train', total_loss, individual_losses)
        
        # Batch log all metrics at once
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    
    
    def validation_step(self, batch, batch_idx):
        wav, adsr_gt = batch
        
        # Ensure wav is on the correct device
        wav = wav.to(self.device)
        adsr_gt = adsr_gt.to(self.device)
        
        mel = self.wav2mel(wav)
        
        # Forward pass
        adsr_pred = self.forward(mel)
        
        # Ultra-efficient loss calculation
        total_loss, individual_losses = self.loss_fn(adsr_pred, adsr_gt)
        
        # Efficient logging
        log_dict = self._prepare_log_dict('val', total_loss, individual_losses)
        
        # Batch log all metrics at once
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.config['lr']
        )
        return optimizer
    
    
    def on_train_epoch_end(self):
        # Log epoch summary
        if self.config.get('wandb_use', False):
            epoch_metrics = {
                'epoch': self.current_epoch,
                'train_loss_epoch': self.trainer.callback_metrics.get('train_loss_epoch', 0),
                'val_loss_epoch': self.trainer.callback_metrics.get('val_loss_epoch', 0),
            }
            
            # Add individual ADSR metrics
            for param_name in self.adsr_params:
                epoch_metrics[f'train_{param_name}_loss_epoch'] = self.trainer.callback_metrics.get(f'train_{param_name}_loss_epoch', 0)
                epoch_metrics[f'val_{param_name}_loss_epoch'] = self.trainer.callback_metrics.get(f'val_{param_name}_loss_epoch', 0)
            
            wandb.log(epoch_metrics)



