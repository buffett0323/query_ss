import os
import random
import librosa
import numpy as np
import scipy.signal
import pretty_midi
import json
import torch
import torch.nn.functional as F

from multiprocessing import Pool
from omegaconf import OmegaConf
from tqdm import tqdm

SAMPLE_RATE = 44100
FILTER_TIME = 4.0


def yaml_config_hook(config_file):
    """
    Load YAML with OmegaConf to support ${variable} interpolation.
    Also supports nested includes via a 'defaults' section.
    """
    # Load main config
    cfg = OmegaConf.load(config_file)

    # Load nested defaults if any (like Hydra-style)
    if "defaults" in cfg:
        for d in cfg.defaults:
            config_dir, cf = d.popitem()
            cf_path = os.path.join(os.path.dirname(config_file), config_dir, f"{cf}.yaml")
            nested_cfg = OmegaConf.load(cf_path)
            cfg = OmegaConf.merge(cfg, nested_cfg)

        del cfg.defaults

    return cfg



def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch



def save_checkpoint(args, iter, wrapper):
    """Save model checkpoint and optimizer state"""
    checkpoint_path = os.path.join(args.ckpt_path, f'checkpoint_{iter}.pt')

    # Save generator
    torch.save({
        'generator_state_dict': wrapper.generator.state_dict(),
        'optimizer_g_state_dict': wrapper.optimizer_g.state_dict(),
        'scheduler_g_state_dict': wrapper.scheduler_g.state_dict(),
        'discriminator_state_dict': wrapper.discriminator.state_dict(),
        'optimizer_d_state_dict': wrapper.optimizer_d.state_dict(),
        'scheduler_d_state_dict': wrapper.scheduler_d.state_dict(),
        'iter': iter
    }, checkpoint_path)

    # Save latest checkpoint by creating a symlink
    latest_path = os.path.join(args.ckpt_path, 'checkpoint_latest.pt')
    if os.path.exists(latest_path):
        os.remove(latest_path)
    os.symlink(checkpoint_path, latest_path)

    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(args, device, iter, wrapper):
    """Load model checkpoint and optimizer state"""
    if iter == -1:
        # Load latest checkpoint
        checkpoint_path = os.path.join(args.ckpt_path, 'checkpoint_latest.pt')
    else:
        # Load specific checkpoint
        checkpoint_path = os.path.join(args.ckpt_path, f'checkpoint_{iter}.pt')

    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load generator
    wrapper.generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)

    # Try to load optimizer states using safe loading
    if safe_load_optimizer_state(wrapper.optimizer_g, checkpoint['optimizer_g_state_dict']):
        print("Successfully loaded generator optimizer state")
    else:
        print("Continuing with fresh generator optimizer state...")

    try:
        wrapper.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        print("Successfully loaded generator scheduler state")
    except (ValueError, KeyError) as e:
        print(f"Warning: Could not load generator scheduler state: {e}")
        print("Continuing with fresh scheduler state...")

    # Load discriminator
    wrapper.discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)

    if safe_load_optimizer_state(wrapper.optimizer_d, checkpoint['optimizer_d_state_dict']):
        print("Successfully loaded discriminator optimizer state")
    else:
        print("Continuing with fresh discriminator optimizer state...")

    try:
        wrapper.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        print("Successfully loaded discriminator scheduler state")
    except (ValueError, KeyError) as e:
        print(f"Warning: Could not load discriminator scheduler state: {e}")
        print("Continuing with fresh scheduler state...")

    print(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint['iter']



def get_timbre_names(path):
    timbres = set()
    for file in tqdm(os.listdir(path)):
        if file.endswith(".wav"):
            tmp_timbre = file.split(".wav")[0].split("_")
            tmp_timbre = "_".join(tmp_timbre[:-1])
            timbres.add(tmp_timbre)

    timbres = list(timbres)
    random.shuffle(timbres)
    with open("info/timbre_names_mixed.txt", "w") as f:
        for timbre in timbres:
            f.write(timbre + "\n")

    print(len(timbres))


def get_midi_names(path, split):
    midis = set()
    for file in tqdm(os.listdir(path)):
        if file.endswith(".wav"):
            midis.add(file.split(".wav")[0].split("_")[-1])

    midis = list(midis)
    random.shuffle(midis)
    with open(f"info/midi_names_mixed_{split}.txt", "w") as f:
        for midi in midis:
            f.write(midi + "\n")

    print(len(midis))



def process_file(file_info, path):
    stem, file = file_info
    file_name = file.split(".wav")[0]
    if file.endswith(".wav"):
        audio, _ = librosa.load(os.path.join(path, stem, file), sr=None)
        audio = audio[:int(FILTER_TIME * SAMPLE_RATE)]

        # Find indices where amplitude exceeds threshold
        envelope = np.abs(audio)
        peaks, _ = scipy.signal.find_peaks(envelope, distance=SAMPLE_RATE // 20)
        if len(peaks) == 0: return (file_name, [])

        # Get peak amplitudes
        peak_amplitudes = envelope[peaks]

        # Get top 10 peaks by amplitude
        if len(peaks) > 10:
            top_indices = np.argsort(peak_amplitudes)[-10:]
            peaks = peaks[top_indices]
            peak_amplitudes = peak_amplitudes[top_indices]

        # Convert peak indices to time and filter peaks > 8 seconds
        peak_info = [peak_idx / SAMPLE_RATE for peak_idx in peaks]

        return (file_name, peak_info) if peak_info else (file_name, [])



def get_onset_from_midi(midi_file):
    """
    Extract onset times from a MIDI file, filtering to only include onsets before 8 seconds.

    Args:
        midi_file: Path to the MIDI file

    Returns:
        onset_times: List of onset times in seconds before 8 seconds
    """
    try:
        # Load MIDI file
        midi_data = pretty_midi.PrettyMIDI(midi_file)

        # Collect all note onset times
        onset_times = []
        n_count = 0
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                n_count += 1
                if note.start <= FILTER_TIME:
                    onset_times.append(note.start)

        return sorted(list(set(onset_times)))

    except Exception as e:
        print(f"Error processing {midi_file}: {e}")
        return []



def get_top_5_peak_position(audio):
    envelope = np.abs(audio)
    peaks, _ = scipy.signal.find_peaks(envelope, distance=SAMPLE_RATE // 20)
    if len(peaks) == 0: return []

    # Get peak amplitudes
    peak_amplitudes = envelope[peaks]

    # Get top 5 peaks by amplitude
    if len(peaks) > 5:
        top_indices = np.argsort(peak_amplitudes)[-5:]
        peaks = peaks[top_indices]

    # Convert peak indices to time in seconds
    peak_times = peaks / SAMPLE_RATE

    # Return as list for JSON compatibility
    return peak_times.tolist()


def process_beatport_file(file_info):
    """
    Process a single beatport file to extract peak positions

    Args:
        file_info: tuple of (file_path, file_name)

    Returns:
        tuple of (file_name, peaks_list)
    """
    file_path, file_name = file_info
    try:
        audio, _ = librosa.load(file_path, sr=None)
        audio = audio[:int(FILTER_TIME * SAMPLE_RATE)]
        peaks = get_top_5_peak_position(audio)
        return (file_name, peaks)
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return (file_name, [])


def safe_load_optimizer_state(optimizer, state_dict):
    """
    Safely load optimizer state dict by matching parameters by name
    rather than by parameter group structure.
    """
    try:
        # First try the standard loading method
        optimizer.load_state_dict(state_dict)
        return True
    except (ValueError, KeyError) as e:
        print(f"Standard optimizer loading failed: {e}")
        print("Attempting parameter-wise loading...")

        # Get current parameter names
        current_params = []
        for group in optimizer.param_groups:
            for param in group['params']:
                current_params.append(param)

        # Create mapping from parameter id to parameter
        if 'state' in state_dict:
            saved_state = state_dict['state']
            current_state = optimizer.state

            # Try to match saved states to current parameters
            # This is a simplified approach - in practice you might want more sophisticated matching
            print("Attempting to restore optimizer state for matching parameters...")
            matched_params = 0

            for param_id, param_state in saved_state.items():
                if param_id < len(current_params):
                    current_state[current_params[param_id]] = param_state
                    matched_params += 1

            print(f"Restored state for {matched_params} parameters")

        return False


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_parameter_count(num_params):
    """Format parameter count in readable format (e.g., 6M, 300M, 4B)."""
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.1f}B"
    elif num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.1f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.1f}K"
    else:
        return str(num_params)


def print_model_info(wrapper):
    """Print information about model parameters."""
    gen_params = count_parameters(wrapper.generator)
    disc_params = count_parameters(wrapper.discriminator)
    total_params = gen_params + disc_params

    print(f"\n{'='*50}")
    print(f"MODEL PARAMETER COUNT")
    print(f"{'='*50}")
    print(f"Generator parameters: {format_parameter_count(gen_params)} ({gen_params:,})")
    print(f"Discriminator parameters: {format_parameter_count(disc_params)} ({disc_params:,})")
    print(f"Total parameters: {format_parameter_count(total_params)} ({total_params:,})")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # get_timbre_names("/home/buffett/dataset/EDM_FAC_DATA/train")
    # get_midi_names("/home/buffett/dataset/EDM_FAC_DATA/train", "train")
    # get_midi_names("/home/buffett/dataset/EDM_FAC_DATA/evaluation", "evaluation")


    # train_path = "/home/buffett/dataset/EDM_FAC_DATA/single_note_midi/train/midi"
    # eval_path = "/home/buffett/dataset/EDM_FAC_DATA/single_note_midi/evaluation/midi"
    # train_midis = []
    # eval_midis = []

    # with open("info/midi_names_mixed_train.txt", "r") as f:
    #     for line in f:
    #         train_midis.append(os.path.join(train_path, line.strip() + ".mid"))

    # with open("info/midi_names_mixed_evaluation.txt", "r") as f:
    #     for line in f:
    #         eval_midis.append(os.path.join(eval_path, line.strip() + ".mid"))

    # onset_records_train = {}
    # onset_records_eval = {}

    # for midi_file in tqdm(train_midis):
    #     m = midi_file.split("/")[-1].split(".mid")[0]
    #     onset_times = get_onset_from_midi(midi_file)
    #     onset_records_train[m] = onset_times

    # for midi_file in tqdm(eval_midis):
    #     m = midi_file.split("/")[-1].split(".mid")[0]
    #     onset_times = get_onset_from_midi(midi_file)
    #     onset_records_eval[m] = onset_times

    # with open("/home/buffett/dataset/EDM_FAC_DATA/json/onset_records_mixed_train.json", "w") as f:
    #     json.dump(onset_records_train, f, indent=4)

    # with open("/home/buffett/dataset/EDM_FAC_DATA/json/onset_records_mixed_evaluation.json", "w") as f:
    #     json.dump(onset_records_eval, f, indent=4)

    beatport_path = "/home/buffett/dataset/EDM_FAC_DATA/beatport/"

    for split in ["evaluation", "train"]:
        split_path = os.path.join(beatport_path, split)

        # Prepare list of files to process
        file_infos = []
        for file in os.listdir(split_path):
            if file.endswith(".wav"):
                file_path = os.path.join(split_path, file)
                file_infos.append((file_path, file))

        print(f"Processing {len(file_infos)} files for {split} split...")

        # Use multiprocessing to process files in parallel
        num_processes = min(16, os.cpu_count())  # Use up to 8 processes or CPU count
        print(f"Using {num_processes} processes")

        with Pool(processes=num_processes) as pool:
            # Process files with progress bar
            results = list(tqdm(
                pool.imap(process_beatport_file, file_infos),
                total=len(file_infos),
                desc=f"Processing {split}"
            ))

        # Convert results to dictionary
        peak_records = {file_name: peaks for file_name, peaks in results}

        # Save results to JSON
        with open(f"/home/buffett/dataset/EDM_FAC_DATA/json/beatport_peak_records_{split}.json", "w") as f:
            json.dump(peak_records, f, indent=4)

        print(f"Saved {len(peak_records)} peak records for {split} split")


def log_rms(wav, hop=512, eps=1e-7):
    """Return log-RMS envelope. wav: (B, 1, T_samples)."""
    rms = torch.sqrt(
        F.avg_pool1d(wav ** 2, kernel_size=hop, stride=hop) + eps
    )
    return torch.log(rms + eps).squeeze(1)        # (B, T_frames)
