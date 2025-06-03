import os
import yaml
import random
import json
import torch
import torchaudio
import librosa
import librosa.display
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from collections import defaultdict
from tqdm import tqdm


def get_top_2_peak_infos(
    load_path,
    segment_duration=0.95,
    sample_rate=16000,
    amp_thres=0.5
):
    # Load waveform
    waveform = np.load(load_path).astype(np.float32)

    # Pretend out of bounds
    segment_samples = int(segment_duration*sample_rate)
    thres = waveform.shape[0] - segment_samples
    cut_waveform = waveform[:thres]

    # Envelope (amplitude)
    envelope = np.abs(cut_waveform)

    # Find peaks
    peaks, _ = scipy.signal.find_peaks(envelope, distance=sample_rate // 20)
    if len(peaks) == 0: return [] # print("No peaks found.")

    # Get peak amplitudes
    peak_amplitudes = envelope[peaks]

    # Filter peaks above threshold
    peak_info = []
    for peak_idx, amp in zip(peaks, peak_amplitudes):
        if amp >= amp_thres:
            time_sec = peak_idx / sample_rate
            peak_info.append((peak_idx, amp, time_sec))

    return peak_info



def find_top_2_peak_segments(
    load_path,
    segment_duration=0.95,
    sample_rate=16000,
    amp_thres=0.5
):
    # Load waveform
    if load_path.endswith(".npy"):
        waveform = np.load(load_path).astype(np.float32)
    elif load_path.endswith(".wav"):
        waveform, sr = torchaudio.load(load_path)
        waveform = waveform.numpy()

        # Resample if needed
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(
                torch.from_numpy(waveform), sr, sample_rate
            ).numpy()

        # Check if waveform is mono
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(axis=0)

    # Pretend out of bounds
    segment_samples = int(segment_duration*sample_rate)
    thres = waveform.shape[0] - segment_samples
    cut_waveform = waveform[:thres]

    # Envelope (amplitude)
    envelope = np.abs(cut_waveform)

    # Find peaks
    peaks, _ = scipy.signal.find_peaks(envelope, distance=sample_rate // 20)
    if len(peaks) == 0: return [] # print("No peaks found.")


    # Get peak amplitudes and top 5
    peak_amplitudes = envelope[peaks]
    top_indices = np.argsort(peak_amplitudes)#[::-1]  # highest to lowest

    top_peaks = peaks[top_indices]
    top_amplitudes = peak_amplitudes[top_indices]
    top_times = top_peaks / sample_rate

    # Sort by time
    sorted_peaks = sorted(zip(top_amplitudes, top_peaks, top_times), key=lambda x: x[0], reverse=True)
    segments = []

    for i, (amp, peak_idx, time_sec) in enumerate(sorted_peaks, 1):
        if amp < amp_thres:
            break

        start_idx = peak_idx
        end_idx = start_idx + segment_samples

        segment = waveform[start_idx:end_idx]
        segments.append(segment)

        # print(f"Peak {i}: Time = {time_sec:.3f}s, Amplitude = {amp:.5f}, Segment shape = {segment.shape}")

    return segments


def plot_waveform(npy_path, savefig_name="../sample_audio/waveform.png"):
    waveform = np.load(npy_path)
    sample_rate = 16000
    if len(waveform.shape) == 1:
        waveform = waveform[np.newaxis, :]

    num_samples = waveform.shape[1]
    duration = num_samples / sample_rate
    time_axis = np.linspace(0, duration, num_samples)

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, waveform[0], label="Channel 1")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(f"Waveform with {duration} seconds")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savefig_name)
    plt.close()
    print("Saved the figure to ", savefig_name)


def npy2audio(npy_path, savefig_name="../sample_audio/temp_audio.wav"):
    waveform = np.load(npy_path)

    # Convert to torch tensor and reshape to [1, num_samples] for mono
    waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    sample_rate = 16000

    # Save as .wav
    torchaudio.save(savefig_name, waveform, sample_rate)
    print("Saved to ", savefig_name)


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg


def define_param_groups(model, weight_decay, optimizer_name):
   def exclude_from_wd_and_adaptation(name):
       if 'bn' in name:
           return True
       if optimizer_name == 'lars' and 'bias' in name:
           return True

   param_groups = [
       {
           'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
           'weight_decay': weight_decay,
           'layer_adaptation': True,
       },
       {
           'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
           'weight_decay': 0.,
           'layer_adaptation': False,
       },
   ]
   return param_groups


# Save to text files
def save_to_txt(filename, data):
    with open(filename, "w") as f:
        for item in data:
            f.write(f"{item}\n")

def load_from_txt(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines()]



def plot_spec_and_save(spectrogram, savefig_name, sr=16000):
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.squeeze().numpy()

    plt.figure(figsize=(6, 6))
    librosa.display.specshow(
        spectrogram,
        x_axis='time',
        y_axis='mel',
        sr=sr,
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel-Spectrogram")
    plt.savefig(savefig_name)


def resize_spec(spectrogram, target_size=(256, 256)):
    """ Resize spectrogram using interpolation to fit Swin Transformer input """
    resized_spec = zoom(spectrogram, (target_size[0] / spectrogram.shape[0], target_size[1] / spectrogram.shape[1]), order=3)
    return resized_spec


def plot_and_save_logmel_spectrogram(x_i, x_j, song_name, output_dir, stage="original", sample_rate=16000):
    """
    Plots and saves the log-mel spectrograms for the transformed, normed, and augmented versions.

    Args:
        x_i (Tensor): Transformed spectrogram of x_i
        x_j (Tensor): Transformed spectrogram of x_j
        song_name (str): Name of the song or track to save the plot
        output_dir (str): Directory to save the plots
        stage (str): A string indicating which stage (original, normed, augmented)
        sample_rate (int): The sample rate of the audio
    """
    # Convert to numpy for plotting (assuming single channel)
    x_i = x_i.cpu().numpy()
    x_j = x_j.cpu().numpy()

    # Create figure and axis
    fig, axes = plt.subplots(2, 1, figsize=(6, 12), dpi=300)
    fig.suptitle(f"Log-Mel Spectrograms for {song_name} ({stage})", fontsize=16)

    # Plot for x_i (e.g., transformed or original)
    cax1 = axes[0].imshow(x_i[0], aspect='auto', origin='lower', cmap='inferno', interpolation='none')
    axes[0].set_title(f"Spectrogram for x_i ({stage})")
    axes[0].set_xlabel("Time (frames)")
    axes[0].set_ylabel("Mel bands")
    fig.colorbar(cax1, ax=axes[0], format="%+2.0f dB")

    # Plot for x_j (e.g., transformed or original)
    cax2 = axes[1].imshow(x_j[0], aspect='auto', origin='lower', cmap='inferno', interpolation='none')
    axes[1].set_title(f"Spectrogram for x_j ({stage})")
    axes[1].set_xlabel("Time (frames)")
    axes[1].set_ylabel("Mel bands")
    fig.colorbar(cax2, ax=axes[1], format="%+2.0f dB")

    # Save figure
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plot_filename = f"{song_name}_{stage}.png"
    plt.savefig(f"{output_dir}/{plot_filename}")
    plt.close()

def plot_mel_spectrogram_librosa(log_mel_spec, output_dir, song_name, stage="original", sample_rate=16000):
    log_mel_spec = log_mel_spec.cpu().numpy().squeeze()

    # Plot Mel Spectrogram
    plt.figure(figsize=(10, 6), dpi=150)
    librosa.display.specshow(log_mel_spec, x_axis='time', y_axis='mel', sr=sample_rate,
                             hop_length=160, n_fft=1024, n_mels=64, fmin=60, fmax=7800, cmap='inferno')
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"{song_name} - {stage} Mel Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Mel Frequency Bands")

    # Save the plot
    plt.tight_layout()
    plot_filename = f"{song_name}_{stage}_mel_spec_librosa.png"
    plt.savefig(f"{output_dir}/{plot_filename}")
    plt.close()


if __name__ == "__main__":
    path = "/mnt/gestalt/home/buffett/beatport_analyze/chorus_audio_16000_095sec_npy_bass_other/0747dfb4-58bf-4be6-9b2e-7b08c69e07df_1/bass_other_seg_0.npy"
    npy2audio(path)
