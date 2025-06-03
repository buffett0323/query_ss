"""
Objective:
Whether the separated "other" stem contains a dominant instrument or useful sound, as opposed to being too noisy, silent, or leakage-heavy.
You want to ensure the "other" stem is not just a cocktail of noise and bleed-over from vocals/drums.
"""


# Steps
# 1. Check RMS Energy
# 2. Check Spectral Flatness


from tqdm import tqdm
import os
import torch
import torchaudio
import numpy as np
from multiprocessing import Pool, cpu_count

def compute_energy_db(x):
    """Compute signal energy in decibels."""
    energy = torch.mean(x ** 2, dim=-1) + 1e-10  # avoid log(0)
    return 10 * torch.log10(energy)

def classify_clean_segments(vocals, bass, drums, other, y_mixture):
    """
    Classify mixture segments as 'clean_target', 'clean_residual', or 'noisy'.

    Parameters:
    - y_mixture: [B, T] waveform tensor
    - separator_model: model that returns (vocals, bass, drums, other)
    - db_threshold: energy difference in dB

    Returns:
    - List of tags per sample: 'clean_target', 'clean_residual', or 'noisy'
    """
    with torch.no_grad():

        # Define target vs residual and convert to mono by averaging channels
        target = other.mean(dim=0) if other.dim() > 1 else other
        residual = (vocals + bass + drums).mean(dim=0) if (vocals + bass + drums).dim() > 1 else (vocals + bass + drums)
        y_mixture = y_mixture.mean(dim=0) if y_mixture.dim() > 1 else y_mixture

        # Compute energies
        E_mix = compute_energy_db(y_mixture)
        E_target = compute_energy_db(target)
        E_residual = compute_energy_db(residual)

        return  E_mix, E_target, E_residual


def process_file(file_path_tuple, db_threshold=20.0):
    """Process a single file and return results"""
    path, file = file_path_tuple

    # load npy files
    y_other = np.load(os.path.join(path, file, "other.npy"))
    y_vocals = np.load(os.path.join(path, file, "vocals.npy"))
    y_drums = np.load(os.path.join(path, file, "drums.npy"))
    y_bass = np.load(os.path.join(path, file, "bass.npy"))
    y_mix = np.load(os.path.join(path, file, "mix.npy"))

    # Convert to torch tensors
    y_mix = torch.from_numpy(y_mix)
    y_vocals = torch.from_numpy(y_vocals)
    y_bass = torch.from_numpy(y_bass)
    y_drums = torch.from_numpy(y_drums)
    y_other = torch.from_numpy(y_other)

    E_mix, E_target, E_residual = classify_clean_segments(y_vocals, y_bass, y_drums, y_other, y_mix)

    # Decision logic
    tags = ""#[]
    if E_mix - E_residual > db_threshold:
        tags = "clean_target"
        print(f"{file} is clean_target")
    elif E_mix - E_target > db_threshold:
        tags = "clean_residual"
    else:
        tags = "noisy" #tags.append("noisy")

    return file, tags, E_mix - E_residual



if __name__ == "__main__":
    path = "/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_npy/"
    sr = 44100
    tags_count = {"clean_target": 0, "clean_residual": 0, "noisy": 0}

    clean_target = []
    clean_target_10 = []

    # Create list of files to process
    files = [(path, file) for file in os.listdir(path)]

    # Use multiprocessing pool
    num_processes = cpu_count() - 1  # Leave one CPU free
    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files)))

    # Process results
    for file, tags, target_diff in results:
        tags_count[tags] += 1

        if tags == "clean_target":
            clean_target.append(file)

        if target_diff > 10:
            clean_target_10.append(file)

    print(tags_count)
    print("clean_target", len(clean_target))
    print("clean_target > 20", len(clean_target_10))

    # Write clean target list to file
    with open("../info/clean_target_files.txt", "w") as f:
        for file in clean_target:
            f.write(f"{file}\n")

    with open("../info/clean_target_20_files.txt", "w") as f:
        for file in clean_target_10:
            f.write(f"{file}\n")
