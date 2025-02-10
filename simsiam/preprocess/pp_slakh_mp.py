import os
import librosa
import torchaudio
import torchaudio.transforms as T
import numpy as np
from tqdm import tqdm
import yaml
import soundfile as sf
from multiprocessing import Pool, cpu_count

# Path
input_path = "/mnt/gestalt/home/ddmanddman/slakh2100_flac_redux/train"
output_path = "/mnt/gestalt/home/ddmanddman/slakh2100_buffett/train"
os.makedirs(output_path, exist_ok=True)

# Number of processes (use all available cores)
NUM_PROCESSES = min(cpu_count(), 8)  # Limit to 8 cores if too many processes slow down disk IO

def convert_flac_to_npy(input_flac, target_sr=16000):
    """ Convert FLAC file to NumPy array with target sample rate. """
    waveform, sr = torchaudio.load(input_flac, format="flac")
    if sr != 44100:
        print("Wrong sr when converting:", sr)    

    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform.numpy()


def energy_activity_detection(audio, output_path, basename, sr=16000):
    """ Perform energy-based activity detection and save a 4s segment. """
    non_silent_intervals = librosa.effects.split(audio, top_db=30)  # Detect non-silent regions
    segment_duration = sr * 4  # 4 seconds in samples
    selected_segment = None

    for start, end in non_silent_intervals:
        if (end - start) >= segment_duration:
            selected_segment = audio[start : start + segment_duration]
            break

    if selected_segment is not None:
        # Save to .npy
        output_npy = os.path.join(output_path, f"{basename}.npy")
        np.save(output_npy, selected_segment)

        # Save to .wav
        output_wav = os.path.join(output_path, f"{basename}.wav")
        sf.write(output_wav, selected_segment, sr)


def process_track(track):
    """ Function to process a single track. This will run in parallel. """
    yaml_file = os.path.join(input_path, track, "metadata.yaml")
    
    if not os.path.exists(yaml_file):
        return None  # Skip if metadata is missing

    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
    
    stem_mapping = {stem: info["inst_class"] for stem, info in data["stems"].items()}
    
    label_set = set()
    
    for stem in os.listdir(os.path.join(input_path, track, "stems")):
        input_flac = os.path.join(input_path, track, "stems", stem)
        output_wav_folder = os.path.join(output_path, track)
        os.makedirs(output_wav_folder, exist_ok=True)

        # Get base name from YAML mapping
        basename = stem_mapping[os.path.splitext(stem)[0]]
        label_set.add(basename)

        # Convert FLAC to NumPy array
        wav_npy = convert_flac_to_npy(input_flac)  # (1, 4082068)

        if len(wav_npy.shape) > 1:
            wav_npy = wav_npy.squeeze(0)

        # Perform energy-based activity detection
        energy_activity_detection(wav_npy, output_wav_folder, basename=basename)
    
    return label_set


if __name__ == "__main__":
    label_set = set()
    tracks = os.listdir(input_path)  # List all tracks in the input path

    # Use multiprocessing Pool to process tracks in parallel
    with Pool(NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(process_track, tracks), total=len(tracks), desc="Processing Tracks"))

    # Collect all unique labels
    for res in results:
        if res is not None:
            label_set.update(res)

    # Save labels to file
    os.makedirs('../info', exist_ok=True)
    with open('../info/slakh_label.txt', 'w') as f:
        for label in sorted(label_set):  # Sort for consistency
            f.write(f"{label}\n")

    print("Processing complete. Labels saved.")
