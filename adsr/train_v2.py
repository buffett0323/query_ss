from byol_a2.common import (np, Path, torch,
     get_logger, load_yaml_config, seed_everything, get_timestamp, hash_text)
from byol_a2.byol_pytorch import BYOL
from byol_a2.models import AudioNTT2022, load_pretrained_weights
from byol_a2.augmentations import NormalizeBatch, PrecomputedNorm
from byol_a2.dataset import ADSRDataset
import pytorch_lightning as pl
import fire
import logging
import nnAudio.features
from tqdm import tqdm
import warnings
import torch.multiprocessing as tmp

# Filter out NVML warning
warnings.filterwarnings("ignore", message="Can't initialize NVML")

def process_batch(batch, to_spec, device):
    wav1, wav2 = batch
    wav1 = wav1.to(device)
    wav2 = wav2.to(device)
    lms1 = (to_spec(wav1) + torch.finfo().eps).log().unsqueeze(1)
    lms2 = (to_spec(wav2) + torch.finfo().eps).log().unsqueeze(1)
    lms_batch = torch.cat([lms1, lms2], dim=0)
    return lms_batch.detach().cpu().numpy()

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
        self.to_spec = nnAudio.features.MelSpectrogram(
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.f_min,
            fmax=cfg.f_max,
            center=True,
            power=2,
            verbose=False,
        )

    def forward(self, images1, images2):
        return self.learner(images1, images2)

    def training_step(self, wavs, batch_idx):
        def to_np(A): return [a.cpu().numpy() for a in A]
        # Convert raw audio into a log-mel spectrogram and pre-normalize it.
        device = self.device
        self.to_spec = self.to_spec.to(device)
        self.learner = self.learner.to(device)

        wav1, wav2 = wavs
        wav1 = wav1.to(device)
        wav2 = wav2.to(device)
        lms1 = (self.to_spec(wav1) + torch.finfo().eps).log().unsqueeze(1)
        lms2 = (self.to_spec(wav2) + torch.finfo().eps).log().unsqueeze(1)
        lms_batch = torch.cat([lms1, lms2], dim=0)
        lms_batch = self.pre_norm(lms_batch)
        lms1 = lms_batch[:lms_batch.shape[0]//2]
        lms2 = lms_batch[lms_batch.shape[0]//2:]
        paired_inputs = (lms1, lms2)

        # Form a batch and post-normalize it.
        bs = paired_inputs[0].shape[0]
        paired_inputs = torch.cat(paired_inputs) # [(B,1,T,F), (B,1,T,F)] -> (2*B,1,T,F)
        mb, sb = to_np((paired_inputs.mean(), paired_inputs.std()))
        paired_inputs = self.post_norm(paired_inputs)
        ma, sa = to_np((paired_inputs.mean(), paired_inputs.std()))

        # Forward to get a loss.
        loss = self.forward(paired_inputs[:bs], paired_inputs[bs:])
        for k, v in {'mb': mb, 'sb': sb, 'ma': ma, 'sa': sa}.items():
            self.log(k, float(v), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_before_zero_grad(self, _):
        self.learner.update_moving_average()

    def calc_norm_stats(self, data_loader, n_stats=10000, device='cuda'):
        # Calculate normalization statistics from the training dataset.
        n_stats = min(n_stats, len(data_loader.dataset))
        logging.info(f'Calculating mean/std using random {n_stats} samples from population {len(data_loader.dataset)} samples...')
        device = self.device
        self.to_spec = self.to_spec.to(device)
        X = []
        for wavs in tqdm(data_loader):
            wav1, wav2 = wavs
            wav1 = wav1.to(device)
            wav2 = wav2.to(device)
            lms1 = (self.to_spec(wav1) + torch.finfo().eps).log().unsqueeze(1)
            lms2 = (self.to_spec(wav2) + torch.finfo().eps).log().unsqueeze(1)
            lms_batch = torch.cat([lms1, lms2], dim=0)
            X.extend([x for x in lms_batch.detach().cpu().numpy()])
            if len(X) >= n_stats: break
        X = np.stack(X)
        norm_stats = np.array([X.mean(), X.std()])
        logging.info(f'  ==> mean/std: {norm_stats}, {norm_stats.shape} <- {X.shape}')
        self.pre_norm = PrecomputedNorm(norm_stats)
        return norm_stats


def complete_cfg(cfg):
    # Set ID.
    cfg.id = (f'AudioNTT2022-BYOLA-{cfg.shape[0]}x{cfg.shape[1]}d{cfg.feature_d}-{get_timestamp()}'
              f'-e{cfg.epochs}b{cfg.bs}l{str(cfg.lr)[2:]}r{cfg.seed}-{hash_text(str(cfg), L=8)}')
    return cfg


def main(config_path='config_v2.yaml') -> None:
    cfg = load_yaml_config(config_path)
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")

    # Verify GPU availability
    if 'cuda' in str(device):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(device)}")

    cfg.unit_samples = int(cfg.sample_rate * cfg.unit_sec)
    complete_cfg(cfg)

    # Essentials
    get_logger(__name__)
    logging.info(cfg)
    seed_everything(cfg.seed)

    # Data preparation
    ds = ADSRDataset(
        data_dir=cfg.data_dir,
        unit_sec=cfg.unit_sec
    )
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.bs,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=cfg.prefetch_factor,
    )

    # Training preparation
    logging.info(f'Training {cfg.id}...')

    # Model
    model = AudioNTT2022(
        n_mels=cfg.n_mels,
        d=cfg.feature_d
    ).to(device)
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
    ).to(device)
    learner.calc_norm_stats(dl, device=device)

    # # Trainer
    # trainer = pl.Trainer(
    #     gpus=cfg.gpus,
    #     max_epochs=cfg.epochs,
    #     weights_summary=None,
    #     accelerator="ddp"
    # )
    # trainer.fit(learner, dl)
    # if trainer.interrupted:
    #     logging.info('Terminated.')
    #     exit(0)
    # # Saving trained weight.
    # to_file = Path(cfg.checkpoint_folder)/(cfg.id+'.pth')
    # to_file.parent.mkdir(exist_ok=True, parents=True)
    # torch.save(model.state_dict(), to_file)
    # logging.info(f'Saved weight as {to_file}')


if __name__ == '__main__':
    fire.Fire(main)
