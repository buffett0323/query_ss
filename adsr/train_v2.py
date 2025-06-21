from byol_a2.common import (np, Path, torch,
     get_logger, load_yaml_config, seed_everything, get_timestamp, hash_text)
from byol_a2.byol_pytorch import BYOL
from byol_a2.models import AudioNTT2022, load_pretrained_weights
from byol_a2.augmentations import NormalizeBatch, PrecomputedNorm
from byol_a2.dataset import ADSRDataset, ADSR_h5_Dataset
import pytorch_lightning as pl
import fire
import logging
import nnAudio.features
import warnings
import torch.multiprocessing as tmp
import wandb
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

# Filter out NVML warning
warnings.filterwarnings("ignore", message="Can't initialize NVML")
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
torch.backends.cudnn.deterministic = False  # Disable for speed


class BYOLALearner(pl.LightningModule):
    """BYOL-A learner. Shows batch statistics for each epochs."""

    def __init__(self, cfg, model, tfms, **kwargs):
        super().__init__()
        self.learner = BYOL(
            model,
            image_size=cfg.shape,
            **kwargs
        )
        self.lr = cfg.lr
        self.tfms = tfms
        self.post_norm = NormalizeBatch()
        # self.to_spec = nnAudio.features.MelSpectrogram(
        #     sr=cfg.sample_rate,
        #     n_fft=cfg.n_fft,
        #     win_length=cfg.win_length,
        #     hop_length=cfg.hop_length,
        #     n_mels=cfg.n_mels,
        #     fmin=cfg.f_min,
        #     fmax=cfg.f_max,
        #     center=True,
        #     power=2,
        #     verbose=False,
        # )

    def forward(self, images1, images2):
        return self.learner(images1, images2)

    def training_step(self, specs, batch_idx):
        def to_np(A): return [a.cpu().numpy() for a in A]

        # Move both specs to device at once and combine operations
        device = self.device
        spec1, spec2 = specs
        spec1, spec2 = spec1.to(device), spec2.to(device)

        # Combine log operations and avoid redundant concatenations
        eps = torch.finfo(spec1.dtype).eps
        lms1 = (spec1 + eps).log()
        lms2 = (spec2 + eps).log()

        # Combine normalization in one operation
        lms_batch = torch.cat([lms1, lms2], dim=0)
        lms_batch = self.pre_norm(lms_batch)

        # Split back efficiently
        bs = lms1.shape[0]
        lms1_norm = lms_batch[:bs]
        lms2_norm = lms_batch[bs:]

        # Post-normalize in one operation
        paired_inputs = torch.cat([lms1_norm, lms2_norm], dim=0)
        mb, sb = to_np((paired_inputs.mean(), paired_inputs.std()))
        paired_inputs = self.post_norm(paired_inputs)
        ma, sa = to_np((paired_inputs.mean(), paired_inputs.std()))

        # Forward pass
        loss = self.forward(paired_inputs[:bs], paired_inputs[bs:])

        # Log statistics
        for k, v in {'mb': mb, 'sb': sb, 'ma': ma, 'sa': sa}.items():
            self.log(k, float(v), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_before_zero_grad(self, _):
        self.learner.update_moving_average()

    def calc_norm_stats(self, mel_mean, mel_std):
        # # Calculate normalization statistics from the training dataset.
        # n_stats = min(n_stats, len(data_loader.dataset))
        # logging.info(f'Calculating mean/std using random {n_stats} samples from population {len(data_loader.dataset)} samples...')
        # device = self.device
        # # self.to_spec = self.to_spec.to(device)
        # X = []
        # for specs in tqdm(data_loader):
        #     spec1, spec2 = specs
        #     spec1 = spec1.to(device)
        #     spec2 = spec2.to(device)
        #     lms1 = (spec1 + torch.finfo().eps).log()#.unsqueeze(1)
        #     lms2 = (spec2 + torch.finfo().eps).log()#.unsqueeze(1)
        #     lms_batch = torch.cat([lms1, lms2], dim=0)
        #     X.extend([x for x in lms_batch.detach().cpu().numpy()])
        #     if len(X) >= n_stats: break
        # X = np.stack(X)
        # norm_stats = np.array([X.mean(), X.std()])
        norm_stats = np.array([mel_mean, mel_std])
        self.pre_norm = PrecomputedNorm(norm_stats)
        return norm_stats


def complete_cfg(cfg):
    # Set ID.
    cfg.id = (f'AudioNTT2022-BYOLA-{cfg.shape[0]}x{cfg.shape[1]}d{cfg.feature_d}-{get_timestamp()}'
              f'-e{cfg.epochs}b{cfg.bs}l{str(cfg.lr)[2:]}r{cfg.seed}-{hash_text(str(cfg), L=8)}')
    return cfg


def main(config_path='config_v2.yaml') -> None:
    cfg = load_yaml_config(config_path)

    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.device_id[0])
        logging.info(f"Using GPU: {torch.cuda.get_device_name(cfg.device_id[0])}")
    else:
        raise RuntimeError("CUDA device requested but not available")

    cfg.unit_samples = int(cfg.sample_rate * cfg.unit_sec)
    complete_cfg(cfg)

    # Essentials
    get_logger(__name__)
    logging.info(cfg)
    seed_everything(cfg.seed)

    # Data preparation
    # ds = ADSRDataset(
    #     metadata_dir=cfg.metadata_dir,
    #     data_dir=cfg.data_dir,
    # )
    ds = ADSR_h5_Dataset(h5_path=cfg.h5_path, cache_size=getattr(cfg, 'cache_size', 1000))
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.bs,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=cfg.prefetch_factor,
        # generator=torch.Generator().manual_seed(cfg.seed),  # Add deterministic shuffling
    )

    # Training preparation
    logging.info(f'Training {cfg.id}...')

    # Model
    model = AudioNTT2022(
        n_mels=cfg.n_mels,
        d=cfg.feature_d
    )
    if cfg.resume is not None:
        load_pretrained_weights(model, cfg.resume)

    # Training
    learner = BYOLALearner(
        cfg, model,
        tfms=None,
        hidden_layer=-1,
        projection_size=cfg.proj_size,
        projection_hidden_size=cfg.proj_dim,
        moving_average_decay=cfg.ema_decay,
    )
    learner.calc_norm_stats(cfg.mel_mean, cfg.mel_std)

    # Initialize wandb
    if cfg.log_wandb:
        project = "byol-a-adsr-v2"
        name = cfg.wandb_name
        save_dir = cfg.save_dir
        wandb_logger = WandbLogger(
            project=project,
            name=name,
            save_dir=save_dir,
            log_model=False,  # Avoid logging full model files to WandB
        )
    else:
        wandb_logger = None

    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.checkpoint_folder),
        filename=f'{cfg.id}-{{epoch:02d}}',
        every_n_epochs=5,  # Reduced from 2 to 5 for less I/O overhead
        save_top_k=-1,  # Save all checkpoints
        save_last=True
    )

    # Trainer
    cb = [TQDMProgressBar(refresh_rate=10)]
    cb.append(checkpoint_callback)
    trainer = pl.Trainer(
        devices=[cfg.device_id[0]],  # Use only GPU 1
        max_epochs=cfg.epochs,
        accelerator="gpu",
        strategy="auto",  # Use auto strategy for single GPU
        logger=wandb_logger,
        callbacks=cb
    )
    trainer.fit(learner, dl)
    if trainer.interrupted:
        logging.info('Terminated.')
        exit(0)

    # Saving final weight.
    to_file = Path(cfg.checkpoint_folder)/(cfg.id+'.pth')
    to_file.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), to_file)
    logging.info(f'Saved weight as {to_file}')

    if wandb_logger is not None:
        wandb_logger.finalize("success")


if __name__ == '__main__':
    fire.Fire(main)
