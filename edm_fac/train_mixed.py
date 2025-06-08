import sys
import warnings
import argparse
import torch
import os
import time
import torch
import logging
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

from dataset import EDM_Paired_Dataset, EDM_Unpaired_Dataset
from utils import yaml_config_hook, get_infinite_loader, save_checkpoint, load_checkpoint
import dac


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Wrapper:
    def __init__(
        self,
        args,
        accelerator,
        val_paired_data,
        val_unpaired_data
    ):
        assert args.max_note - args.min_note + 1 == 88, "Pitch numbers must be 88"
        self.generator = dac.model.MyDAC(
            encoder_dim=args.encoder_dim,
            encoder_rates=args.encoder_rates,
            latent_dim=args.latent_dim,
            decoder_dim=args.decoder_dim,
            decoder_rates=args.decoder_rates,
            sample_rate=args.sample_rate,
            timbre_classes=args.timbre_classes,
            pitch_nums=args.max_note - args.min_note + 1, # 88
        ).to(accelerator.device)

        self.optimizer_g = torch.optim.AdamW(self.generator.parameters(), lr=args.base_lr)
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_g, gamma=1.0)

        self.discriminator = dac.model.Discriminator().to(accelerator.device)
        self.optimizer_d = torch.optim.AdamW(self.discriminator.parameters(), lr=args.base_lr)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_d, gamma=1.0)

        # Losses
        self.stft_loss = MultiScaleSTFTLoss().to(accelerator.device)
        # self.mel_loss = MelSpectrogramLoss().to(accelerator.device) # Change it to original state
        self.mel_loss = MelSpectrogramLoss(
            n_mels=[5, 10, 20, 40, 80, 160, 320],
            window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
            mel_fmin=[0, 0, 0, 0, 0, 0, 0],
            mel_fmax=[None, None, None, None, None, None, None],
            pow=1.0,
            mag_weight=0.0,
        ).to(accelerator.device)
        self.l1_loss = L1Loss().to(accelerator.device)
        self.gan_loss = GANLoss(discriminator=self.discriminator).to(accelerator.device)

        # ✅ Switched both to CrossEntropyLoss for balanced classification
        self.timbre_loss = nn.CrossEntropyLoss().to(accelerator.device)
        self.content_loss = nn.CrossEntropyLoss().to(accelerator.device)

        # Loss lambda parameters
        self.params = {
            "gen/mel-loss": 15.0,
            "adv/loss_feature": 2.0,
            "adv/loss_g": 1.0,
            "vq/commitment_loss": 0.25,
            "vq/codebook_loss": 1.0,
            "pred/timbre_loss": 1.0,
            "pred/content_loss": 10.0,
        }

        self.val_paired_data = val_paired_data
        self.val_unpaired_data = val_unpaired_data

    @staticmethod
    def timbre_acc(pred_logits, target_labels):
        """
        pred_logits: [B, num_classes]
        target_labels: [B]
        """
        preds = torch.argmax(pred_logits, dim=-1)
        correct = (preds == target_labels).sum().item()
        total = target_labels.size(0)
        acc = correct / total
        return acc


@torch.no_grad()
def save_samples(args, accelerator, tracker_step, wrapper):
    """ Paired data validation """
    samples = [wrapper.val_paired_data[idx] for idx in args.val_idx]
    batch = wrapper.val_paired_data.collate(samples)
    batch = util.prepare_batch(batch, accelerator.device)

    out = wrapper.generator(
        audio_data=batch['target'].audio_data,
        content_match=batch['content_match'].audio_data,
        timbre_match=batch['timbre_converted'].audio_data,
    )

    recons = AudioSignal(out["audio"].cpu(), args.sample_rate)
    recons_gt = AudioSignal(batch['target'].audio_data.cpu(), args.sample_rate)
    ref_content = AudioSignal(batch['content_match'].audio_data.cpu(), args.sample_rate)
    ref_timbre = AudioSignal(batch['timbre_converted'].audio_data.cpu(), args.sample_rate)

    os.makedirs(os.path.join(args.save_path, 'sample_audio', f'iter_{tracker_step}'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'sample_audio', f'iter_{tracker_step}', 'conv'), exist_ok=True)
    

    # Conversion
    for i, sample_idx in enumerate(args.val_idx):
        single_recon = AudioSignal(recons.audio_data[i], args.sample_rate)
        single_recon_gt = AudioSignal(recons_gt.audio_data[i], args.sample_rate)
        single_ref_content = AudioSignal(ref_content.audio_data[i], args.sample_rate)
        single_ref_timbre = AudioSignal(ref_timbre.audio_data[i], args.sample_rate)

        recon_path = os.path.join(args.save_path, 'sample_audio', f'iter_{tracker_step}', 'conv', f'{sample_idx}_recon.wav')
        recon_gt_path = os.path.join(args.save_path, 'sample_audio', f'iter_{tracker_step}', 'conv', f'{sample_idx}_gt.wav')
        ref_content_path = os.path.join(args.save_path, 'sample_audio', f'iter_{tracker_step}', 'conv', f'{sample_idx}_ref_content.wav')
        ref_timbre_path = os.path.join(args.save_path, 'sample_audio', f'iter_{tracker_step}', 'conv', f'{sample_idx}_ref_timbre.wav')

        single_recon.write(recon_path)
        single_recon_gt.write(recon_gt_path)
        single_ref_content.write(ref_content_path)
        single_ref_timbre.write(ref_timbre_path)
        
        
    """ Unpaired data validation """
    samples = [wrapper.val_unpaired_data[idx] for idx in args.val_idx]
    batch = wrapper.val_unpaired_data.collate(samples)
    batch = util.prepare_batch(batch, accelerator.device)
    
    out = wrapper.generator.forward_unpaired(
        audio_data=batch['target'].audio_data,
    )
    
    recons = AudioSignal(out["audio"].cpu(), args.sample_rate)
    recons_gt = AudioSignal(batch['target'].audio_data.cpu(), args.sample_rate)
    
    os.makedirs(os.path.join(args.save_path, 'sample_audio', f'iter_{tracker_step}', 'recon'), exist_ok=True)
    
    # Reconstruction
    for i, sample_idx in enumerate(args.val_idx):
        single_recon = AudioSignal(recons.audio_data[i], args.sample_rate)
        single_recon_gt = AudioSignal(recons_gt.audio_data[i], args.sample_rate)

        recon_path = os.path.join(args.save_path, 'sample_audio', f'iter_{tracker_step}', 'recon', f'{sample_idx}_recon.wav')
        recon_gt_path = os.path.join(args.save_path, 'sample_audio', f'iter_{tracker_step}', 'recon', f'{sample_idx}_gt.wav')

        single_recon.write(recon_path)
        single_recon_gt.write(recon_gt_path)
    
    
    


def main(args, accelerator):

    device = accelerator.device
    util.seed(args.seed)
    print(f"Using device: {device}")

    # Checkpoint direction
    os.makedirs(args.ckpt_path, exist_ok=True)


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
    train_paired_data = EDM_Paired_Dataset(
        root_path=args.root_path,
        midi_path=args.midi_path,
        data_path=args.data_path,
        duration=args.duration,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        min_note=args.min_note,
        max_note=args.max_note,
        split="train",
    )
    
    train_unpaired_data = EDM_Unpaired_Dataset(
        beatport_path=args.beatport_path,
        data_path=args.data_path,
        duration=args.duration,
        total_duration=args.total_duration,
        sample_rate=args.sample_rate,
        split="train",
    )

    val_paired_data = EDM_Paired_Dataset(   
        root_path=args.root_path,
        midi_path=args.midi_path,
        data_path=args.data_path,
        duration=args.duration,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        min_note=args.min_note,
        max_note=args.max_note,
        split="evaluation",
    )
    
    val_unpaired_data = EDM_Unpaired_Dataset(
        beatport_path=args.beatport_path,
        data_path=args.data_path,
        duration=args.duration,
        total_duration=args.total_duration,
        sample_rate=args.sample_rate,
        split="evaluation",
    )
    wrapper = Wrapper(args, accelerator, val_paired_data, val_unpaired_data)


    # Load checkpoint if exists
    if args.resume:
        start_iter = load_checkpoint(args, device, -1, wrapper) or 0
        tracker.step = start_iter
    else:
        tracker.step = 0


    # Accelerate dataloaders
    train_paired_loader = accelerator.prepare_dataloader(
        train_paired_data,
        start_idx=tracker.step * args.batch_size,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        collate_fn=train_paired_data.collate,
    )
    train_unpaired_loader = accelerator.prepare_dataloader(
        train_unpaired_data,
        start_idx=tracker.step * args.batch_size,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        collate_fn=train_unpaired_data.collate,
    )
    train_paired_loader = get_infinite_loader(train_paired_loader)
    train_unpaired_loader = get_infinite_loader(train_unpaired_loader)
    
    val_paired_loader = accelerator.prepare_dataloader(
        val_paired_data,
        start_idx=0,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        collate_fn=val_paired_data.collate,
    )
    val_unpaired_loader = accelerator.prepare_dataloader(
        val_unpaired_data,
        start_idx=0,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        collate_fn=val_unpaired_data.collate,
    )

    
    # Trackers settings
    global train_step, validate, save_checkpoint, save_samples
    train_step = tracker.log("Train", "value", history=False)(
        tracker.track("Train", args.num_iters, completed=tracker.step)(train_step)
    )
    # train_step_paired = tracker.log("Train/Paired", "value", history=False)(
    #     tracker.track("Train/Paired", args.num_iters, completed=tracker.step)(train_step_paired)
    # )
    # train_step_unpaired = tracker.log("Train/Unpaired", "value", history=False)(
    #     tracker.track("Train/Unpaired", args.num_iters, completed=tracker.step)(train_step_unpaired)
    # )
    validate = tracker.track("Validation", int(args.num_iters / args.validate_interval))(validate)
    save_checkpoint = when(lambda: accelerator.local_rank == 0)(save_checkpoint)
    save_samples = when(lambda: accelerator.local_rank == 0)(save_samples)

    
    # Loop
    with tracker.live:
        for tracker.step, paired_batch in enumerate(train_paired_loader, start=tracker.step):
            unpaired_batch = next(train_unpaired_loader)
            train_step(args, accelerator, paired_batch, unpaired_batch, wrapper)

            # Save Checkpoint
            if tracker.step % args.save_interval == 0:
                save_checkpoint(args, tracker.step, wrapper)

            # Validation
            if tracker.step % args.validate_interval == 0:
                validate(args, accelerator, val_paired_loader, val_unpaired_loader, wrapper)

            # Save validation samples
            if tracker.step % args.sample_freq == 0:
                save_samples(args, accelerator, tracker.step, wrapper)

            if tracker.step == args.num_iters:
                break


# @timer
@torch.no_grad()
def validate(args, accelerator, val_paired_loader, val_unpaired_loader, wrapper):
    output = {}
    for i, paired_batch in enumerate(val_paired_loader):
        output = validate_step_paired(args, accelerator, paired_batch, wrapper)
        if i >= args.validate_steps:
            break
    
    for i, unpaired_batch in enumerate(val_unpaired_loader):
        output = validate_step_unpaired(args, accelerator, unpaired_batch, wrapper)
        if i >= args.validate_steps:
            break
        
        
    if hasattr(wrapper.optimizer_g, "consolidate_state_dict"):
        wrapper.optimizer_g.consolidate_state_dict()
        wrapper.optimizer_d.consolidate_state_dict()

    return output


# @timer
@torch.no_grad()
def validate_step_paired(args, accelerator, batch, wrapper):
    wrapper.generator.eval()
    wrapper.discriminator.eval()
    batch = util.prepare_batch(batch, accelerator.device)

    target_audio = batch['target']
    with torch.no_grad():
        out = wrapper.generator(
            audio_data=target_audio.audio_data,
            content_match=batch['content_match'].audio_data,
            timbre_match=batch['timbre_converted'].audio_data,
        )
    output = {}
    recons = AudioSignal(out["audio"], args.sample_rate)
    output["gen/stft-loss"] = wrapper.stft_loss(recons, target_audio)
    output["gen/mel-loss"] = wrapper.mel_loss(recons, target_audio)
    output["gen/l1-loss"] = wrapper.l1_loss(recons, target_audio)

    # Timbre prediction loss and accuracy
    # output["pred/timbre_loss"] = wrapper.timbre_loss(out["pred_timbre_id"], batch['timbre_id'])
    # output["pred/content_loss"] = wrapper.content_loss(out["pred_pitch"], batch['pitch'])
    # output["pred/timbre_acc"] = wrapper.timbre_acc(out["pred_timbre_id"], batch['timbre_id'])
    # output["pred/content_acc"] = wrapper.content_acc(out["pred_pitch"], batch['pitch'])

    return {k: v for k, v in sorted(output.items())}


# @timer
@torch.no_grad()
def validate_step_unpaired(args, accelerator, batch, wrapper):
    wrapper.generator.eval()
    wrapper.discriminator.eval()
    batch = util.prepare_batch(batch, accelerator.device)

    target_audio = batch['target']

    with torch.no_grad():
        out = wrapper.generator.forward_unpaired(
            audio_data=target_audio.audio_data,
        )
    output = {}
    recons = AudioSignal(out["audio"], args.sample_rate)
    output["gen/stft-loss"] = wrapper.stft_loss(recons, target_audio)
    output["gen/mel-loss"] = wrapper.mel_loss(recons, target_audio)
    output["gen/l1-loss"] = wrapper.l1_loss(recons, target_audio)

    # Timbre prediction loss and accuracy
    # output["pred/timbre_loss"] = wrapper.timbre_loss(out["pred_timbre_id"], batch['timbre_id'])
    # output["pred/content_loss"] = wrapper.content_loss(out["pred_pitch"], batch['pitch'])
    # output["pred/timbre_acc"] = wrapper.timbre_acc(out["pred_timbre_id"], batch['timbre_id'])
    # output["pred/content_acc"] = wrapper.content_acc(out["pred_pitch"], batch['pitch'])

    return {k: v for k, v in sorted(output.items())}


# @timer
def train_step(args, accelerator, paired_batch, unpaired_batch, wrapper):
    output = {}
    output.update(train_step_paired(args, accelerator, paired_batch, wrapper))
    output.update(train_step_unpaired(args, accelerator, unpaired_batch, wrapper))
    return output


# @timer
def train_step_paired(args, accelerator, batch, wrapper):
    train_start_time = time.time()
    wrapper.generator.train()
    wrapper.discriminator.train()

    # Load Batch Items
    batch = util.prepare_batch(batch, accelerator.device)
    target_audio = batch['target']

    # DAC Model
    with accelerator.autocast():
        out = wrapper.generator(
            audio_data=target_audio.audio_data,
            content_match=batch['content_match'].audio_data,
            timbre_match=batch['timbre_converted'].audio_data,
        )
        output = {}
        recons = AudioSignal(out["audio"], args.sample_rate)

        # Discriminator Losses
        output["adv/disc_loss"] = wrapper.gan_loss.discriminator_loss(recons, target_audio)

    wrapper.optimizer_d.zero_grad()
    accelerator.backward(output["adv/disc_loss"])
    accelerator.scaler.unscale_(wrapper.optimizer_d)
    grad_norm_d = torch.nn.utils.clip_grad_norm_(wrapper.discriminator.parameters(), 10.0)
    accelerator.step(wrapper.optimizer_d)
    wrapper.scheduler_d.step()

    # Generator Losses
    with accelerator.autocast():
        output["gen/stft-loss"] = wrapper.stft_loss(recons, target_audio)
        output["gen/mel-loss"] = wrapper.mel_loss(recons, target_audio)
        output["gen/l1-loss"] = wrapper.l1_loss(recons, target_audio)

        output["adv/loss_g"], output["adv/loss_feature"] = wrapper.gan_loss.generator_loss(recons, target_audio)
        output["vq/commitment_loss"] = out["vq/commitment_loss"]
        output["vq/codebook_loss"] = out["vq/codebook_loss"]

        # # Added predictor losses
        # output["pred/timbre_loss"] = wrapper.timbre_loss(out["pred_timbre_id"], batch['timbre_id'])
        # output["pred/content_loss"] = wrapper.content_loss(out["pred_pitch"], batch['pitch'])

        # Total Loss
        output["loss_gen_all"] = sum([v * output[k] for k, v in wrapper.params.items() if k in output])

    # Optimizer
    wrapper.optimizer_g.zero_grad()
    accelerator.backward(output["loss_gen_all"])
    accelerator.scaler.unscale_(wrapper.optimizer_g)
    grad_norm_g = torch.nn.utils.clip_grad_norm_(wrapper.generator.parameters(), 1000.0)
    accelerator.step(wrapper.optimizer_g)
    wrapper.scheduler_g.step()
    accelerator.update()

    # Logging
    # output["other/learning_rate"] = wrapper.optimizer_g.param_groups[0]["lr"]
    # output["other/grad_norm_g"] = grad_norm_g
    # output["other/grad_norm_d"] = grad_norm_d
    # output["other/time_per_step"] = time.time() - train_start_time
    
    return {k: v for k, v in sorted(output.items())}


# @timer
def train_step_unpaired(args, accelerator, batch, wrapper):

    train_start_time = time.time()
    wrapper.generator.train()
    wrapper.discriminator.train()

    # Load Batch Items
    batch = util.prepare_batch(batch, accelerator.device)
    target_audio = batch['target']

    # DAC Model
    with accelerator.autocast():
        out = wrapper.generator.forward_unpaired(
            audio_data=target_audio.audio_data,
        )
        output = {}
        recons = AudioSignal(out["audio"], args.sample_rate)

        # Discriminator Losses
        output["adv/disc_loss"] = wrapper.gan_loss.discriminator_loss(recons, target_audio)

    wrapper.optimizer_d.zero_grad()
    accelerator.backward(output["adv/disc_loss"])
    accelerator.scaler.unscale_(wrapper.optimizer_d)
    grad_norm_d = torch.nn.utils.clip_grad_norm_(wrapper.discriminator.parameters(), 10.0)
    accelerator.step(wrapper.optimizer_d)
    wrapper.scheduler_d.step()

    
    # Generator Losses
    with accelerator.autocast():
        output["gen/stft-loss"] = wrapper.stft_loss(recons, target_audio)
        output["gen/mel-loss"] = wrapper.mel_loss(recons, target_audio)
        output["gen/l1-loss"] = wrapper.l1_loss(recons, target_audio)

        output["adv/loss_g"], output["adv/loss_feature"] = wrapper.gan_loss.generator_loss(recons, target_audio)
        output["vq/commitment_loss"] = out["vq/commitment_loss"]
        output["vq/codebook_loss"] = out["vq/codebook_loss"]

        # # Added predictor losses
        # output["pred/timbre_loss"] = wrapper.timbre_loss(out["pred_timbre_id"], batch['timbre_id'])
        # output["pred/content_loss"] = wrapper.content_loss(out["pred_pitch"], batch['pitch'])

        # Total Loss
        output["loss_gen_all"] = sum([v * output[k] for k, v in wrapper.params.items() if k in output])

    # Optimizer
    wrapper.optimizer_g.zero_grad()
    accelerator.backward(output["loss_gen_all"])
    accelerator.scaler.unscale_(wrapper.optimizer_g)
    grad_norm_g = torch.nn.utils.clip_grad_norm_(wrapper.generator.parameters(), 1000.0)
    accelerator.step(wrapper.optimizer_g)
    wrapper.scheduler_g.step()
    accelerator.update()

    # Logging
    # output["other/learning_rate"] = wrapper.optimizer_g.param_groups[0]["lr"]
    # output["other/grad_norm_g"] = grad_norm_g
    # output["other/grad_norm_d"] = grad_norm_d
    # output["other/time_per_step"] = time.time() - train_start_time

    return {k: v for k, v in sorted(output.items())}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDM-FAC")

    config = yaml_config_hook("configs/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.ckpt_path, exist_ok=True)

    # Initialize accelerator
    accelerator = ml.Accelerator()
    if accelerator.local_rank != 0:
        sys.tracebacklimit = 0
    main(args, accelerator)
