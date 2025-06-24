import torch
import torch.nn.functional as F
import wandb
import nnAudio.features
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import AudioADSRDataset
from model import ASTWithProjectionHead
from adsr_spv.not_used.config import TrainConfig


def train(cfg: TrainConfig):
    # Create checkpoint directory
    checkpoint_dir = Path(cfg.wandb_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if cfg.wandb_use:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_name,
            dir=cfg.wandb_dir,
            config=cfg.__dict__
        )

    train_ds = AudioADSRDataset(
        data_dir=cfg.data_dir,
        split="train",
    )
    val_ds = AudioADSRDataset(
        data_dir=cfg.data_dir,
        split="val",
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )
    # test_ds = AudioADSRDataset(
    #     data_dir=cfg.data_dir,
    #     split="test",
    # )
    # test_dl = DataLoader(
    #     test_ds,
    #     batch_size=cfg.batch_size,
    #     shuffle=False,
    #     num_workers=cfg.num_workers
    # )

    model = ASTWithProjectionHead(
        d_model=cfg.d_model,
        d_out=cfg.d_out,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        patch_size=cfg.patch_size,
        patch_stride=cfg.patch_stride,
        input_channels=cfg.input_channels,
    ).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    to_spec = nnAudio.features.MelSpectrogram(
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    ).to(cfg.device)

    # Training
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(cfg.epochs):
        # Training phase
        model.train()
        running = {"param": 0.0}
        running_adsr = {"attack": 0.0, "decay": 0.0, "sustain": 0.0, "release": 0.0}

        for wav, adsr_gt in tqdm(train_dl, desc=f"Training Epoch {epoch}"):
            wav = wav.to(cfg.device)
            adsr_gt = adsr_gt.to(cfg.device)

            mel = to_spec(wav).unsqueeze(1) # [BS, 1, 128, 256]

            # Forward pass
            adsr_pred = model(mel)

            # Parameter loss (supervised)
            loss = F.mse_loss(adsr_pred, adsr_gt)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running["param"] += loss.item()

            # Calculate individual ADSR losses
            for i, param_name in enumerate(["attack", "decay", "sustain", "release"]):
                param_loss = F.mse_loss(adsr_pred[:, i], adsr_gt[:, i])
                running_adsr[param_name] += param_loss.item()

            # Log step metrics
            if cfg.wandb_use:
                log_dict = {
                    "step": global_step,
                    "step_param_loss": loss.item(),
                }
                # Add individual ADSR parameter losses
                for param_name in ["attack", "decay", "sustain", "release"]:
                    log_dict[f"step_{param_name}_loss"] = running_adsr[param_name] / (global_step + 1)
                wandb.log(log_dict)
            global_step += 1

        # Calculate average losses
        avg_param_loss = running["param"]/len(train_dl)
        avg_adsr_losses = {param: loss/len(train_dl) for param, loss in running_adsr.items()}

        # Validation phase
        model.eval()
        val_running = {"param": 0.0}
        val_running_adsr = {"attack": 0.0, "decay": 0.0, "sustain": 0.0, "release": 0.0}

        with torch.no_grad():
            for wav, adsr_gt in tqdm(val_dl, desc=f"Validation Epoch {epoch}"):
                wav = wav.to(cfg.device)
                adsr_gt = adsr_gt.to(cfg.device)

                mel = to_spec(wav).unsqueeze(1)
                adsr_pred = model(mel)

                # Validation loss
                val_loss = F.mse_loss(adsr_pred, adsr_gt)
                val_running["param"] += val_loss.item()

                # Calculate individual ADSR validation losses
                for i, param_name in enumerate(["attack", "decay", "sustain", "release"]):
                    param_loss = F.mse_loss(adsr_pred[:, i], adsr_gt[:, i])
                    val_running_adsr[param_name] += param_loss.item()

        avg_val_loss = val_running["param"]/len(val_dl)
        avg_val_adsr_losses = {param: loss/len(val_dl) for param, loss in val_running_adsr.items()}

        # Log epoch metrics to wandb
        if cfg.wandb_use:
            log_dict = {
                "epoch": epoch,
                "train_param_loss": avg_param_loss,
                "val_param_loss": avg_val_loss,
            }
            # Add individual ADSR parameter losses for both train and validation
            for param_name in ["attack", "decay", "sustain", "release"]:
                log_dict[f"train_{param_name}_loss"] = avg_adsr_losses[param_name]
                log_dict[f"val_{param_name}_loss"] = avg_val_adsr_losses[param_name]
            wandb.log(log_dict)

        print(f"Epoch {epoch:02d} – Train param loss: {avg_param_loss:.4f}, Val param loss: {avg_val_loss:.4f}")
        print(f"  Train ADSR losses - A: {avg_adsr_losses['attack']:.4f}, D: {avg_adsr_losses['decay']:.4f}, S: {avg_adsr_losses['sustain']:.4f}, R: {avg_adsr_losses['release']:.4f}")
        print(f"  Val ADSR losses   - A: {avg_val_adsr_losses['attack']:.4f}, D: {avg_val_adsr_losses['decay']:.4f}, S: {avg_val_adsr_losses['sustain']:.4f}, R: {avg_val_adsr_losses['release']:.4f}")

        # Save checkpoint every 20 epochs
        if (epoch + 1) % cfg.save_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'train_loss': avg_param_loss,
                'val_loss': avg_val_loss,
                'config': cfg.__dict__
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'train_loss': avg_param_loss,
                'val_loss': avg_val_loss,
                'config': cfg.__dict__
            }, best_model_path)
            print(f"Best model saved: {best_model_path}")


# -----------------------------------------------------------------------------
#  Entry‑point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)
