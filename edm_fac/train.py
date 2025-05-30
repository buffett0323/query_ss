import sys
import warnings
import argparse
import torch
import os
import time
import torch
import logging
import wandb
warnings.simplefilter('ignore')

from modules.commons import *
from losses import *
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from dac.nn.loss import MultiScaleSTFTLoss, MelSpectrogramLoss, GANLoss, L1Loss
from audiotools import AudioSignal
from audiotools import ml
from audiotools.ml.decorators import Tracker, timer, when
from audiotools.core import util

from dataset import EDM_Render_Dataset
from utils import yaml_config_hook, get_infinite_loader, save_checkpoint, load_checkpoint
import dac

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Wrapper:
    def __init__(
        self,
        args,
        accelerator,
        val_data
    ):
        self.generator = dac.model.MyDAC(
            timbre_classes=284,
            encoder_dim=args.encoder_dim,
            encoder_rates=args.encoder_rates,
            latent_dim=args.latent_dim,
            decoder_dim=args.decoder_dim,
            decoder_rates=args.decoder_rates,
            sample_rate=args.sample_rate,
        ).to(accelerator.device)
        self.optimizer_g = torch.optim.AdamW(self.generator.parameters(), lr=args.base_lr)
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_g, gamma=1.0)

        self.discriminator = dac.model.Discriminator().to(accelerator.device)
        self.optimizer_d = torch.optim.AdamW(self.discriminator.parameters(), lr=args.base_lr)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_d, gamma=1.0)
        
        # Losses
        self.stft_loss = MultiScaleSTFTLoss().to(accelerator.device)
        self.mel_loss = MelSpectrogramLoss(
            n_mels=[5, 10, 20, 40, 80, 160, 320],
            window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
            mel_fmin=[0, 0, 0, 0, 0, 0, 0],
            mel_fmax=[None, None, None, None, None, None, None],
            pow=1.0,
            mag_weight=0.0,
            clamp_eps=1e-5,
        ).to(accelerator.device)
        self.l1_loss = L1Loss().to(accelerator.device)
        self.gan_loss = GANLoss(discriminator=self.discriminator).to(accelerator.device)
        self.content_loss = FocalLoss(gamma=2).to(accelerator.device)

        # Loss parameters
        self.params = {
            "gen/mel-loss": 15.0,
            "adv/loss_feature": 2.0,
            "adv/loss_g": 1.0,
            "vq/commitment_loss": 0.25,
            "vq/codebook_loss": 1.0,
        }
        
        self.val_data = val_data
        

@torch.no_grad()
def save_samples(args, accelerator, tracker_step, wrapper):
    samples = [wrapper.val_data[idx] for idx in args.val_idx]
    batch = wrapper.val_data.collate(samples)
    batch = util.prepare_batch(batch, accelerator.device)

    out = wrapper.generator(
        audio_data=batch['input'].audio_data,
        content_match=batch['content_match'].audio_data,
        timbre_match=batch['timbre_match'].audio_data,
    )
    
    recons = AudioSignal(out["audio"].cpu(), args.sample_rate)
    inputs = AudioSignal(batch['input'].audio_data.cpu(), args.sample_rate)
    os.makedirs(os.path.join(args.save_path, 'sample_audio', f'iter_{tracker_step}'), exist_ok=True)

    for i, sample_idx in enumerate(args.val_idx):
        single_recon = AudioSignal(recons.audio_data[i], args.sample_rate)
        single_input = AudioSignal(inputs.audio_data[i], args.sample_rate)
        
        recon_path = os.path.join(args.save_path, 'sample_audio', f'iter_{tracker_step}', f'recon_{sample_idx}.wav')
        input_path = os.path.join(args.save_path, 'sample_audio', f'iter_{tracker_step}', f'orig_{sample_idx}.wav')
        
        single_recon.write(recon_path)
        single_input.write(input_path)


def main(args, accelerator):
    
    device = accelerator.device
    util.seed(args.seed)
    print(f"Using device: {device}")
    
    # Initialize wandb
    if args.log_wandb and accelerator.local_rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            dir=args.wandb_dir
        )
    
    Path(args.save_path).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(args.save_path, 'sample_audio')).mkdir(exist_ok=True, parents=True)
    tracker = Tracker(
        writer=(
            SummaryWriter(log_dir=f"{args.save_path}/logs") 
                if accelerator.local_rank == 0 else None
        ), 
        log_file=f"{args.save_path}/log.txt", 
        rank=accelerator.local_rank,
    )

    # Build datasets and dataloaders
    train_data = EDM_Render_Dataset(
        root_path=args.root_path,
        midi_path=args.midi_path, 
        duration=args.duration,
        sample_rate=args.sample_rate,
        min_note=args.min_note,
        max_note=args.max_note,
        stems=args.stems,
        split="train"
    )
    
    val_data = EDM_Render_Dataset(
        root_path=args.root_path,
        midi_path=args.midi_path, 
        duration=args.duration,
        sample_rate=args.sample_rate,
        min_note=args.min_note,
        max_note=args.max_note,
        stems=args.stems,
        split="evaluation"
    )
    wrapper = Wrapper(args, accelerator, val_data)
    
    
    # Load checkpoint if exists
    start_iter = load_checkpoint(args, device, -1, wrapper) or 0
    tracker.step = start_iter
    
    
    # Accelerate dataloaders
    train_loader = accelerator.prepare_dataloader(
        train_data,
        start_idx=tracker.step * args.batch_size,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        collate_fn=train_data.collate,
    )
    train_loader = get_infinite_loader(train_loader)
    val_loader = accelerator.prepare_dataloader(
        val_data,
        start_idx=0,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        collate_fn=val_data.collate,
    )
    
    # Trackers settings
    global train_step, validate, save_checkpoint, save_samples
    train_step = tracker.log("train", "value", history=False)(
        tracker.track("train", args.num_iters, completed=tracker.step)(train_step)
    )
    validate = tracker.track("val", len(val_loader))(validate)
    save_checkpoint = when(lambda: accelerator.local_rank == 0)(save_checkpoint)
    save_samples = when(lambda: accelerator.local_rank == 0)(save_samples)
    
    # Loop
    with tracker.live:
        for tracker.step, batch in enumerate(train_loader, start=tracker.step):
            train_step(args, accelerator, batch, tracker.step, wrapper)
            
            if tracker.step % args.save_interval == 0:
                save_checkpoint(args, tracker.step, wrapper)
            
            if tracker.step % args.validate_interval == 0:
                validate(args, accelerator, val_loader, wrapper)
            
            if tracker.step % args.sample_freq == 0:
                save_samples(args, accelerator, tracker.step, wrapper)    
            
                
            if tracker.step == args.num_iters:
                break

    # Finish wandb run
    if accelerator.local_rank == 0:
        try:
            wandb.finish()
        except:
            pass

# @timer
@torch.no_grad()
def validate(args, accelerator, val_loader, wrapper):
    output = {}
    for batch in val_loader:
        output = validate_step(args, accelerator, batch, wrapper)
    
    if hasattr(wrapper.optimizer_g, "consolidate_state_dict"):
        print("Consolidating state dicts")
        wrapper.optimizer_g.consolidate_state_dict()
        wrapper.optimizer_d.consolidate_state_dict()
    
    # Log validation metrics to wandb
    try:
        import wandb
        if wandb.run is not None and args.log_wandb:
            wandb.log({f"val/{k}": v.item() if torch.is_tensor(v) else v for k, v in output.items()})
    except:
        pass  # Silently continue if wandb logging fails
    
    return output
        

# @timer
@torch.no_grad()
def validate_step(args, accelerator, batch, wrapper):
    wrapper.generator.eval()
    wrapper.discriminator.eval()
    batch = util.prepare_batch(batch, accelerator.device)
    
    input_audio = batch['input']

    with torch.no_grad():
        out = wrapper.generator(
            audio_data=input_audio.audio_data,
            content_match=batch['content_match'].audio_data,
            timbre_match=batch['timbre_match'].audio_data,
        )
    output = {}        
    recons = AudioSignal(out["audio"], args.sample_rate)
    output["gen/stft-loss"] = wrapper.stft_loss(recons, input_audio)
    output["gen/mel-loss"] = wrapper.mel_loss(recons, input_audio)
    output["gen/l1-loss"] = wrapper.l1_loss(recons, input_audio)    
    return output

# @timer
def train_step(args, accelerator, batch, iter, wrapper):

    train_start_time = time.time()
    wrapper.generator.train()
    wrapper.discriminator.train()
    
    # Load Batch Items
    batch = util.prepare_batch(batch, accelerator.device)
    
    input_audio = batch['input']

    # DAC Model
    with accelerator.autocast():
        out = wrapper.generator(
            audio_data=input_audio.audio_data,
            content_match=batch['content_match'].audio_data,
            timbre_match=batch['timbre_match'].audio_data,
        )
        output = {}        
        recons = AudioSignal(out["audio"], args.sample_rate)

        # Discriminator Losses
        output["adv/disc_loss"] = wrapper.gan_loss.discriminator_loss(recons, input_audio)

    wrapper.optimizer_d.zero_grad()
    accelerator.backward(output["adv/disc_loss"])
    grad_norm_d = torch.nn.utils.clip_grad_norm_(wrapper.discriminator.parameters(), 10.0)
    wrapper.optimizer_d.step()
    wrapper.scheduler_d.step()
    
    # Generator Losses
    with accelerator.autocast():
        output["gen/stft-loss"] = wrapper.stft_loss(recons, input_audio)
        output["gen/mel-loss"] = wrapper.mel_loss(recons, input_audio)
        output["gen/l1-loss"] = wrapper.l1_loss(recons, input_audio)
        
        output["adv/loss_g"], output["adv/loss_feature"] = wrapper.gan_loss.generator_loss(recons, input_audio)
        output["vq/commitment_loss"] = out["vq/commitment_loss"]
        output["vq/codebook_loss"] = out["vq/codebook_loss"]
        
        # Total Loss
        output["loss_gen_all"] = sum([v * output[k] for k, v in wrapper.params.items() if k in output])

    # Optimizer
    wrapper.optimizer_g.zero_grad()
    accelerator.backward(output["loss_gen_all"])
    grad_norm_g = torch.nn.utils.clip_grad_norm_(wrapper.generator.parameters(), 1000.0)
    wrapper.optimizer_g.step()
    wrapper.scheduler_g.step()

    # Logging
    # output["other/learning_rate"] = wrapper.optimizer_g.param_groups[0]["lr"]
    # output["other/batch_size"] = input_audio.batch_size
    output["other/grad_norm_g"] = grad_norm_g
    output["other/grad_norm_d"] = grad_norm_d
    output["other/time_per_step"] = time.time() - train_start_time

    # Log to wandb
    if iter % args.log_interval == 0:  # Log every 10 steps to avoid too frequent logging
        try:
            import wandb
            if wandb.run is not None and args.log_wandb:
                wandb.log({f"train/{k}": v.item() if torch.is_tensor(v) else v for k, v in output.items()}, step=iter)
        except:
            pass  # Silently continue if wandb logging fails

    return {k: v for k, v in sorted(output.items())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDM-FAC")

    config = yaml_config_hook("configs/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()


    # Initialize accelerator
    accelerator = ml.Accelerator()
    if accelerator.local_rank != 0:
        sys.tracebacklimit = 0
    main(args, accelerator)