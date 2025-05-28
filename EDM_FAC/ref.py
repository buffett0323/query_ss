import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools import ml
from audiotools.core import util
from audiotools.data import transforms
from audiotools.data.datasets import AudioDataset
from audiotools.data.datasets import AudioLoader
from audiotools.data.datasets import ConcatDataset
from audiotools.ml.decorators import timer
from audiotools.ml.decorators import Tracker
from audiotools.ml.decorators import when
from torch.utils.tensorboard import SummaryWriter
from dataset import EGDBDisentangledDataset
from torch import nn

import dac
import random
warnings.filterwarnings("ignore", category=UserWarning)

# Enable cudnn autotuner to speed up training
# (can be altered by the funcs.seed function)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
# Uncomment to trade memory for speed.

# Optimizers
AdamW = argbind.bind(torch.optim.AdamW, "generator", "discriminator")
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)


@argbind.bind("generator", "discriminator")
def ExponentialLR(optimizer, gamma: float = 1.0):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


# Models
DAC = argbind.bind(dac.model.DAC)
Discriminator = argbind.bind(dac.model.Discriminator)

# Data
AudioDataset = argbind.bind(AudioDataset, "train", "val")
AudioLoader = argbind.bind(AudioLoader, "train", "val")

# Transforms
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)

# Loss
filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
losses = argbind.bind_module(dac.nn.loss, filter_fn=filter_fn)


def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["Identity"],):
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    transform = transforms.Compose(preprocess, augment, postprocess)
    return transform


@argbind.bind("train", "val", "test")
def build_dataset(
    sample_rate: int,
    duration: float = 0.38,
    root_path: str = None,
    midi_path: str = None,
    n_examples: int = None,
    use_matched_dataset: bool = False,):
    if root_path is None:
        raise ValueError("root_path must be provided for matched dataset")
    dataset = EGDBDisentangledDataset(
        root_path=root_path,
        midi_path=midi_path,
        duration=duration,
        sample_rate=sample_rate,
        transform=build_transform(),
        n_examples=n_examples
    )

    return dataset


@dataclass
class State:
    generator: DAC
    optimizer_g: AdamW
    scheduler_g: ExponentialLR

    discriminator: Discriminator
    optimizer_d: AdamW
    scheduler_d: ExponentialLR

    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    gan_loss: losses.GANLoss
    waveform_loss: losses.L1Loss

    # Update classification losses for timbre
    pitch_loss: nn.BCEWithLogitsLoss
    timbre_loss: nn.CrossEntropyLoss

    train_data: EGDBDisentangledDataset
    val_data: EGDBDisentangledDataset

    tracker: Tracker


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    resume: bool = False,
    tag: str = "latest",
    load_weights: bool = False,):
    generator, g_extra = None, {}
    discriminator, d_extra = None, {}

    if resume:
        kwargs = {
            "folder": f"{save_path}/{tag}",
            "map_location": "cpu",
            "package": not load_weights,
        }
        tracker.print(f"Resuming from {str(Path('.').absolute())}/{kwargs['folder']}")
        if (Path(kwargs["folder"]) / "dac").exists():
            generator, g_extra = DAC.load_from_folder(**kwargs)
        if (Path(kwargs["folder"]) / "discriminator").exists():
            discriminator, d_extra = Discriminator.load_from_folder(**kwargs)

    generator = DAC() if generator is None else generator
    discriminator = Discriminator() if discriminator is None else discriminator

    tracker.print(generator)
    tracker.print(discriminator)

    generator = accel.prepare_model(generator)
    generator.EXTERN.append('librosa.filters')
    discriminator = accel.prepare_model(discriminator)

    with argbind.scope(args, "generator"):
        optimizer_g = AdamW(generator.parameters(), use_zero=accel.use_ddp)
        scheduler_g = ExponentialLR(optimizer_g)
    with argbind.scope(args, "discriminator"):
        optimizer_d = AdamW(discriminator.parameters(), use_zero=accel.use_ddp)
        scheduler_d = ExponentialLR(optimizer_d)

    if "optimizer.pth" in g_extra:
        optimizer_g.load_state_dict(g_extra["optimizer.pth"])
    if "scheduler.pth" in g_extra:
        scheduler_g.load_state_dict(g_extra["scheduler.pth"])
    if "tracker.pth" in g_extra:
        tracker.load_state_dict(g_extra["tracker.pth"])

    if "optimizer.pth" in d_extra:
        optimizer_d.load_state_dict(d_extra["optimizer.pth"])
    if "scheduler.pth" in d_extra:
        scheduler_d.load_state_dict(d_extra["scheduler.pth"])

    sample_rate = accel.unwrap(generator).sample_rate
    with argbind.scope(args, "train"):
        train_data = build_dataset(
            sample_rate=sample_rate,
            duration=args.get("train/AudioDataset.duration", 0.38),
            root_path=args.get("train/build_dataset.root_path"),
            midi_path=args.get("train/build_dataset.midi_path"),
            n_examples=None,
            use_matched_dataset=args.get("train/build_dataset.use_matched_dataset", False)
        )
    with argbind.scope(args, "val"):
        val_data = build_dataset(
            sample_rate=sample_rate,
            duration=args.get("val/AudioDataset.duration", 0.5),
            root_path=args.get("val/build_dataset.root_path"),
            midi_path=args.get("val/build_dataset.midi_path"),
            n_examples=args.get("val/build_dataset.n_examples", None),
            use_matched_dataset=args.get("val/build_dataset.use_matched_dataset", False)
        )

    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    gan_loss = losses.GANLoss(discriminator)

    # Initialize classification losses
    pitch_loss = nn.BCEWithLogitsLoss()
    timbre_loss = nn.CrossEntropyLoss()

    return State(
        generator=generator,
        optimizer_g=optimizer_g,
        scheduler_g=scheduler_g,
        discriminator=discriminator,
        optimizer_d=optimizer_d,
        scheduler_d=scheduler_d,
        waveform_loss=waveform_loss,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        gan_loss=gan_loss,
        pitch_loss=pitch_loss,
        timbre_loss=timbre_loss,
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
    )


@timer()
@torch.no_grad()
def val_loop(batch, state, accel):
    state.generator.eval()
    batch = util.prepare_batch(batch, accel.device)
    signal_gt = batch["input"]
    timbre_gt = batch["timbre_id"]
    di_gt = batch["di_id"]
    tone_gt = batch["tone_id"]
    pitch_gt = batch["pitch"]

    # Only use signal_gt mode for validation (matching inference scenario)
    out = state.generator(
        signal_gt.audio_data,
        signal_gt.audio_data,
        signal_gt.audio_data,
        signal_gt.audio_data,
        signal_gt.sample_rate
    )
    recons = AudioSignal(out["audio"], signal_gt.sample_rate)

    # Calculate losses
    return {
        "mel/loss": state.mel_loss(recons, signal_gt),
        "stft/loss": state.stft_loss(recons, signal_gt),
        "waveform/loss": state.waveform_loss(recons, signal_gt),
        # "pred_timbre/loss": state.timbre_loss(out["pred_timbre"], timbre_gt),
        "pred_di/loss": state.timbre_loss(out["pred_di"], di_gt),
        "pred_tone/loss": state.timbre_loss(out["pred_tone"], tone_gt),
        "pred_pitch/loss": state.pitch_loss(out["pred_pitch"], pitch_gt)
    }


@timer()
def train_loop(state, batch, accel, lambdas, gradient_accumulation_steps=1):
    state.generator.train()
    state.discriminator.train()
    output = {}

    batch = util.prepare_batch(batch, accel.device)
    with torch.no_grad():
        signal_gt = batch["input"] # x (p0, d0, t0)
        ori_di = batch["ori_di"] # x (p0, d0)
        content_match_data = batch["content_match"] # x (p0, d1, t1)
        di_match_data = batch["di_match"] # x (p1, d0, t2)
        tone_match_data = batch["tone_match"] # x (p2, d2, t0)
        di_gt = batch["di_id"] # d0
        tone_gt = batch["tone_id"] # t0
        pitch_gt = batch["pitch"] # p0
        conversion_match = batch["conversion_match"] # x (p0, d2, t2)
        conversion_di_gt = batch['tone_match_di_id'] # d2
        conversion_tone_gt = batch['di_match_tone_id'] # t2

    mode = random.choice(["reconstruction", "conversion", 'removal'])
    if mode == "conversion":
        target_signal = conversion_match
    elif mode == "removal":
        target_signal = ori_di
    else:
        target_signal = signal_gt

    if mode == "reconstruction":
        with accel.autocast():
            out = state.generator(
                signal_gt.audio_data,
                content_match_data.audio_data, # x (p0, d1, t1)
                di_match_data.audio_data, # x (p1, d0, t2)
                tone_match_data.audio_data, # x (p2, d2, t0)
                signal_gt.sample_rate
            )

            recons = AudioSignal(out["audio"], signal_gt.sample_rate)
            # pred_timbre = out["pred_timbre"]
            pred_di = out["pred_di"]
            pred_tone = out["pred_tone"]
            pred_pitch = out["pred_pitch"]
            commitment_loss = out["vq/commitment_loss"]
            codebook_loss = out["vq/codebook_loss"]
    elif mode == "removal":
        with accel.autocast():
            out = state.generator(
                signal_gt.audio_data,
                content_match_data.audio_data, # x (p0, d1, t1)
                di_match_data.audio_data, # x (p1, d0, t2)
                tone_match_data.audio_data, # x (p2, d2, t0)
                signal_gt.sample_rate,
                None,
                mask_tone=True)

            recons = AudioSignal(out["audio"], signal_gt.sample_rate)
            # pred_timbre = out["pred_timbre"]
            pred_di = out["pred_di"]
            pred_tone = out["pred_tone"]
            pred_pitch = out["pred_pitch"]
            commitment_loss = out["vq/commitment_loss"]
            codebook_loss = out["vq/codebook_loss"]
    else:
        with accel.autocast():
            out = state.generator(
                signal_gt.audio_data,
                content_match_data.audio_data, # x (p0, d1, t1)
                tone_match_data.audio_data, # x (p2, d2, t0)
                di_match_data.audio_data, # x (p1, d0, t2)
                signal_gt.sample_rate
            )

            recons = AudioSignal(out["audio"], signal_gt.sample_rate) # x (p0, d2, t2)
            # pred_timbre = out["pred_timbre"]
            pred_di = out["pred_di"]
            pred_tone = out["pred_tone"]
            pred_pitch = out["pred_pitch"]
            commitment_loss = out["vq/commitment_loss"]
            codebook_loss = out["vq/codebook_loss"]

    with accel.autocast():
        # Calculate discriminator loss using ground truth signal (input)
        output["adv/disc_loss"] = state.gan_loss.discriminator_loss(recons, target_signal)

    # Scale discriminator loss by gradient accumulation steps
    output["adv/disc_loss"] = output["adv/disc_loss"] / gradient_accumulation_steps

    state.optimizer_d.zero_grad()
    accel.backward(output["adv/disc_loss"])
    accel.scaler.unscale_(state.optimizer_d)
    output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(
        state.discriminator.parameters(), 10.0
    )
    accel.step(state.optimizer_d)
    state.scheduler_d.step()

    if mode == "reconstruction":
        with accel.autocast():
            # Calculate reconstruction losses using ground truth signal (input)
            output["stft/loss"] = state.stft_loss(recons, target_signal)
            output["mel/loss"] = state.mel_loss(recons, target_signal)
            output["waveform/loss"] = state.waveform_loss(recons, target_signal)
            (
                output["adv/gen_loss"],
                output["adv/feat_loss"],
            ) = state.gan_loss.generator_loss(recons, target_signal)
            output["vq/commitment_loss"] = commitment_loss
            output["vq/codebook_loss"] = codebook_loss
            # output["pred_timbre/loss"] = state.timbre_loss(pred_timbre, timbre_gt)
            output["pred_di/loss"] = state.timbre_loss(pred_di, di_gt)
            output["pred_tone/loss"] = state.timbre_loss(pred_tone, tone_gt)
            output["pred_pitch/loss"] = state.pitch_loss(pred_pitch, pitch_gt)

            # Add classification losses to total loss
            output["loss"] = sum([
                v * output[k] for k, v in lambdas.items() if k in output
            ])
    elif mode == "conversion":
        with accel.autocast():
            # Calculate reconstruction losses using ground truth signal (input)
            output["stft/loss"] = state.stft_loss(recons, target_signal)
            output["mel/loss"] = state.mel_loss(recons, target_signal)
            output["waveform/loss"] = state.waveform_loss(recons, target_signal)
            (
                output["adv/gen_loss"],
                output["adv/feat_loss"],
            ) = state.gan_loss.generator_loss(recons, target_signal)
            output["vq/commitment_loss"] = commitment_loss
            output["vq/codebook_loss"] = codebook_loss
            # output["pred_timbre/loss"] = state.timbre_loss(pred_timbre, target_timbre_id)
            output["pred_di/loss"] = state.timbre_loss(pred_di, conversion_di_gt)
            output["pred_tone/loss"] = state.timbre_loss(pred_tone, conversion_tone_gt)
            output["pred_pitch/loss"] = state.pitch_loss(pred_pitch, pitch_gt)

            # Add classification losses to total loss
            output["loss"] = sum([
                v * output[k] for k, v in lambdas.items() if k in output
            ])
    else:
        with accel.autocast():
            # Calculate reconstruction losses using ground truth signal (input)
            output["stft/loss"] = state.stft_loss(recons, target_signal)
            output["mel/loss"] = state.mel_loss(recons, target_signal)
            output["waveform/loss"] = state.waveform_loss(recons, target_signal)
            (
                output["adv/gen_loss"],
                output["adv/feat_loss"],
            ) = state.gan_loss.generator_loss(recons, target_signal)
            output["vq/commitment_loss"] = commitment_loss
            output["vq/codebook_loss"] = codebook_loss
            # output["pred_timbre/loss"] = state.timbre_loss(pred_timbre, di_timbre_id)
            output["pred_di/loss"] = state.timbre_loss(pred_di, di_gt)
            output["pred_tone/loss"] = state.timbre_loss(pred_tone, tone_gt)
            output["pred_pitch/loss"] = state.pitch_loss(pred_pitch, pitch_gt)

            # Add classification losses to total loss
            output["loss"] = sum([
                v * output[k] for k, v in lambdas.items() if k in output
            ])

    # Scale generator loss by gradient accumulation steps
    output["loss"] = output["loss"] / gradient_accumulation_steps

    state.optimizer_g.zero_grad()
    accel.backward(output["loss"])
    accel.scaler.unscale_(state.optimizer_g)
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.generator.parameters(), 1e3
    )
    accel.step(state.optimizer_g)
    state.scheduler_g.step()
    accel.update()

    # Scale metrics back up for logging
    for k in output:
        if k.startswith(("adv/", "loss")):
            output[k] = output[k] * gradient_accumulation_steps

    output["other/learning_rate"] = state.optimizer_g.param_groups[0]["lr"]
    output["other/batch_size"] = signal_gt.batch_size * accel.world_size

    return {k: v for k, v in sorted(output.items())}


def checkpoint(state, save_iters, save_path):
    metadata = {"logs": state.tracker.history}

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", "mel/loss"):
        state.tracker.print(f"Best generator so far")
        tags.append("best")
    if state.tracker.step in save_iters:
        tags.append(f"{state.tracker.step // 1000}k")

    for tag in tags:
        generator_extra = {
            "optimizer.pth": state.optimizer_g.state_dict(),
            "scheduler.pth": state.scheduler_g.state_dict(),
            "tracker.pth": state.tracker.state_dict(),
            "metadata.pth": metadata,
        }
        accel.unwrap(state.generator).metadata = metadata
        accel.unwrap(state.generator).save_to_folder(
            f"{save_path}/{tag}", generator_extra
        )
        discriminator_extra = {
            "optimizer.pth": state.optimizer_d.state_dict(),
            "scheduler.pth": state.scheduler_d.state_dict(),
        }
        accel.unwrap(state.discriminator).save_to_folder(
            f"{save_path}/{tag}", discriminator_extra
        )


@torch.no_grad()
def save_samples(state, val_idx, writer):
    state.tracker.print("Saving audio samples to TensorBoard")
    state.generator.eval()

    samples = [state.val_data[idx] for idx in val_idx]
    batch = state.val_data.collate(samples)
    batch = util.prepare_batch(batch, accel.device)

    # Get input signal and its matches
    signal_gt = batch["input"]
    content_match_data = batch["content_match"]
    di_match_data = batch["di_match"]
    tone_match_data = batch["tone_match"]

    # Generate reconstruction
    out = state.generator(
        signal_gt.audio_data,
        content_match_data.audio_data,
        di_match_data.audio_data,
        tone_match_data.audio_data,
        signal_gt.sample_rate
    )
    recons = AudioSignal(out["audio"], signal_gt.sample_rate)

    audio_dict = {
        "recons": recons
    }
    if state.tracker.step == 0:
        audio_dict["ground_truth"] = signal_gt

    for k, v in audio_dict.items():
        for nb in range(v.batch_size):
            v[nb].cpu().write_audio_to_tb(
                f"{k}/sample_{nb}.wav", writer, state.tracker.step
            )


def validate(state, val_dataloader, accel):
    for batch in val_dataloader:
        output = val_loop(batch, state, accel)

    # Consolidate state dicts if using ZeroRedundancyOptimizer
    if hasattr(state.optimizer_g, "consolidate_state_dict"):
        state.optimizer_g.consolidate_state_dict()
        state.optimizer_d.consolidate_state_dict()

    # Return mean of all validation metrics
    return output


@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "ckpt",
    num_iters: int = 250000,
    save_iters: list = [10000, 50000, 100000, 200000],
    sample_freq: int = 10000,
    valid_freq: int = 1000,
    batch_size: int = 12,
    val_batch_size: int = 10,
    num_workers: int = 8,
    val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7],
    gradient_accumulation_steps: int = 1,  # Number of steps to accumulate gradients
    lambdas: dict = {
        "mel/loss": 15.0,
        "adv/feat_loss": 2.0,
        "adv/gen_loss": 1.0,
        "vq/commitment_loss": 0.25,
        "vq/codebook_loss": 1.0,
        "pred_timbre/loss": 1.0,
        "pred_di/loss": 1.0,
        "pred_tone/loss": 1.0,
        "pred_pitch/loss": 10.0
    },):

    util.seed(seed)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    writer = (
        SummaryWriter(log_dir=f"{save_path}/logs") if accel.local_rank == 0 else None
    )
    tracker = Tracker(
        writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank
    )

    state = load(args, accel, tracker, save_path)
    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.tracker.step * batch_size,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
    )
    train_dataloader = get_infinite_loader(train_dataloader)
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Wrap the functions so that they neatly track in TensorBoard + progress bars
    # and only run when specific conditions are met.
    global train_loop, val_loop, validate, save_samples, checkpoint
    train_loop = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_loop)
    )
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)

    # These functions run only on the 0-rank process
    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)

    with tracker.live:
        for tracker.step, batch in enumerate(train_dataloader, start=tracker.step):
            train_loop(state, batch, accel, lambdas, gradient_accumulation_steps)

            last_iter = (
                tracker.step == num_iters - 1 if num_iters is not None else False
            )
            if tracker.step % sample_freq == 0 or last_iter:
                save_samples(state, val_idx, writer)

            if tracker.step % valid_freq == 0 or last_iter:
                validate(state, val_dataloader, accel)
                checkpoint(state, save_iters, save_path)
                # Reset validation progress bar, print summary since last validation.
                tracker.done("val", f"Iteration {tracker.step}")

            if last_iter:
                break


if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            train(args, accel)
