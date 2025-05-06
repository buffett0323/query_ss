import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import torchaudio
from multiprocessing import Pool, cpu_count

SEGMENT_TIME = 0.95
SAMPLE_RATE = 16000
AMP_THRES = 0.8 # 0.5
SAVE_DIR = "/mnt/gestalt/home/buffett/beatport_analyze/chorus_bass_other_16000_npy"
FILE_NAME = "bass_other"
os.makedirs(SAVE_DIR, exist_ok=True)


def process_file(file_name):
    file_path = os.path.join(data_dir, file_name, f"{FILE_NAME}.wav")
    waveform, sr = torchaudio.load(file_path)
    
    # Resample if needed
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sr, SAMPLE_RATE
        )

    # Convert to numpy and make mono if needed
    waveform = waveform.numpy()
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(axis=0)
    
    # Save as npy file
    save_path = os.path.join(SAVE_DIR, file_name)
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, f"{FILE_NAME}.npy"), waveform)


if __name__ == "__main__":
    data_dir = "/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_wav"
    
    with open("../info/chorus_audio_16000_npy_track_list.txt", "r") as f:
        track_list = [line.strip() for line in f.readlines()]#[:5]

    # Use multiprocessing to parallelize the processing
    with Pool(cpu_count() - 1) as pool:
        list(tqdm(
            pool.imap(process_file, track_list),
            total=len(track_list),
            desc="Processing files"
        ))