import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import AudioADSRDataset
from model import ASTWithProjectionHead
from config import TrainConfig



def train(cfg: TrainConfig):
    # Initialize wandb
    if cfg.wandb_use:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_name,
            dir=cfg.wandb_dir,
            config=cfg.__dict__
        )

    ds = AudioADSRDataset(
        metadata_dir=Path("/mnt/gestalt/home/buffett/rendered_adsr_dataset_npy"),
        data_dir=Path("/mnt/gestalt/home/buffett/rendered_adsr_dataset_npy_new_mel"),
    )
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )

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

    # Training
    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        running = {"param": 0.0}

        for mel, adsr_gt in tqdm(dl):
            mel = mel.unsqueeze(1) # [1, 1, 128, 256]
            mel = mel.to(cfg.device)
            adsr_gt = adsr_gt.to(cfg.device)

            adsr_pred = model(mel)

            # Parameter loss (supervised)
            loss = F.mse_loss(adsr_pred, adsr_gt)

            # loss = loss_param + cfg.spectral_weight * loss_spec
            opt.zero_grad()
            loss.backward()
            opt.step()

            running["param"] += loss.item()
            # running["spec"] += loss_spec.item()

            # Log step metrics
            if cfg.wandb_use:
                wandb.log({
                    "step": global_step,
                    "step_param_loss": loss.item(),
                })
            global_step += 1

        # Calculate average losses
        avg_param_loss = running["param"]/len(dl)
        # avg_spec_loss = running["spec"]/max(1,len(dl))

        # Log epoch metrics to wandb
        if cfg.wandb_use:
            wandb.log({
                "epoch": epoch,
                "param_loss": avg_param_loss,
                # "spec_loss": avg_spec_loss
            })

        print(f"Epoch {epoch:02d} – param loss {avg_param_loss:.4f}  ")

        # TODO: validation + checkpoint saving


# -----------------------------------------------------------------------------
#  Entry‑point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)
