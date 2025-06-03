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



def find_top_2_peak_segments(
    npy_path,
    segment_duration=0.95,
    sample_rate=16000,
    amp_thres=0.5
):
    # Load waveform
    waveform = np.load(npy_path).astype(np.float32)

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
        if segment.shape[0] < segment_samples:
            print("Length not enough", start_idx, end_idx, waveform.shape)
            continue
        segments.append(segment)

        # print(f"Peak {i}: Time = {time_sec:.3f}s, Amplitude = {amp:.5f}, Segment shape = {segment.shape}")

    return segments


def plot_waveform(npy_path, savefig_name="sample_audio/waveform.png"):
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


def npy2audio(npy_path, savefig_name="sample_audio/temp_audio.wav"):
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


def train_test_split_BPDataset(path="/home/buffett/NAS_NTU/beatport_analyze/verse_audio_16000_npy"):
    path_file = os.listdir(path)
    random.shuffle(path_file)

    # Compute split sizes
    total_files = len(path_file)
    train_size = int(total_files * 9 / 10)
    valid_size = int(total_files * 0.5 / 10)
    test_size = total_files - train_size - valid_size  # Ensure all files are allocated

    # Split dataset
    train_files = path_file[:train_size]
    valid_files = path_file[train_size:train_size + valid_size]
    test_files = path_file[train_size + valid_size:]

    save_to_txt("info/train_bp_verse_8secs.txt", train_files)
    save_to_txt("info/valid_bp_verse_8secs.txt", valid_files)
    save_to_txt("info/test_bp_verse_8secs.txt", test_files)
    print(f"Dataset split complete: {train_size} train, {valid_size} valid, {test_size} test")


def split_dataset_by_song(path="/home/buffett/NAS_NTU/beatport_analyze/verse_audio_16000_npy"):
    path_file = os.listdir(path)
    random.shuffle(path_file)

    song_dict = defaultdict(lambda: "train")
    song_counter_dict = defaultdict(lambda: [])
    unique_songs = set()

    for file in path_file:
        song_name, song_num = file.split("_")[0], int(file.split("_")[-1])
        if song_name == "ae7633e8-27df-4980-812c-9c6dacfb1d22":
            print(song_num)
        unique_songs.add(song_name)
        song_counter_dict[song_name].append(song_num)


    # Compute split sizes
    total_files = len(unique_songs)
    train_size = int(total_files * 9 / 10)
    valid_size = int(total_files * 0.5 / 10)
    test_size = total_files - train_size - valid_size  # Ensure all files are allocated
    print("Size check: ", "train", train_size, "valid", valid_size, "test", test_size)


    # Split dataset by song name
    for i, song_name in tqdm(enumerate(unique_songs)):
        if i < test_size:
            song_dict[song_name] = "test"
        elif i >= test_size and i < test_size + valid_size:
            song_dict[song_name] = "valid"
        else:
            song_dict[song_name] = "train"


    # Add song number to the song name
    train_files, valid_files, test_files = [], [], []

    for song_name, song_split in tqdm(song_dict.items()):
        if song_split == "train":
            for i in song_counter_dict[song_name]:
                train_files.append(f"{song_name}_{i}")
        elif song_split == "valid":
            for i in song_counter_dict[song_name]:
                valid_files.append(f"{song_name}_{i}")
        else:
            for i in song_counter_dict[song_name]:
                test_files.append(f"{song_name}_{i}")


    random.shuffle(train_files)
    random.shuffle(valid_files)
    random.shuffle(test_files)

    save_to_txt("info/train_by_song_name_4secs.txt", train_files)
    save_to_txt("info/valid_by_song_name_4secs.txt", valid_files)
    save_to_txt("info/test_by_song_name_4secs.txt", test_files)
    print(f"Dataset split complete: {len(train_files)} train, {len(valid_files)} valid, {len(test_files)} test")



def split_dataset_by_segment_dict(path="info/chorus_audio_16000_095sec_npy_seg_counter.json"):
    with open(path, "r") as f:
        seg_counter = json.load(f)
        bp_listdir = list(seg_counter.keys())
        random.shuffle(bp_listdir)

    # Compute split sizes
    total_files = len(bp_listdir)
    train_size = int(total_files * 9 / 10)
    valid_size = int(total_files * 0.5 / 10)
    test_size = total_files - train_size - valid_size  # Ensure all files are allocated
    print("Size check: ", "train", train_size, "valid", valid_size, "test", test_size)


    # Split into train/valid/test
    train_files = bp_listdir[:train_size]
    valid_files = bp_listdir[train_size:train_size+valid_size]
    test_files = bp_listdir[train_size+valid_size:]

    # Create split dictionaries
    train_dict = {k: seg_counter[k] for k in train_files}
    valid_dict = {k: seg_counter[k] for k in valid_files}
    test_dict = {k: seg_counter[k] for k in test_files}

    # Save to json files
    with open("info/train_seg_counter.json", "w") as f:
        json.dump(train_dict, f, indent=4)
    with open("info/valid_seg_counter.json", "w") as f:
        json.dump(valid_dict, f, indent=4)
    with open("info/test_seg_counter.json", "w") as f:
        json.dump(test_dict, f, indent=4)

    print(f"Dataset split complete: {len(train_dict)} train, {len(valid_dict)} valid, {len(test_dict)} test")


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



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == "__main__":
    # train_test_split_BPDataset()
    # split_dataset_by_song(path="/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_16000_4secs_npy")
    # split_dataset_by_segment_dict()
    for i in range(10):
        print(random.sample(range(10), 2))
