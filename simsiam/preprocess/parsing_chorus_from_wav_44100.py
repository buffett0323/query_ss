import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from utils import find_top_2_peak_segments, plot_waveform, npy2audio
from multiprocessing import Pool, cpu_count

SEGMENT_TIME = 0.95
SAMPLE_RATE = 16000
AMP_THRES = 0.8 # 0.5
SAVE_DIR = "/mnt/gestalt/home/buffett/beatport_analyze/chorus_audio_16000_095sec_npy_bass_other"
FILE_NAME = "bass_other.wav"
os.makedirs(SAVE_DIR, exist_ok=True)


def process_file(file_name):
    file_path = os.path.join(data_dir, file_name, FILE_NAME)
    segments = find_top_2_peak_segments(
        file_path,
        segment_duration=SEGMENT_TIME,
        sample_rate=SAMPLE_RATE,
        amp_thres=AMP_THRES
    )

    if len(segments) >= 2:
        os.makedirs(os.path.join(SAVE_DIR, file_name), exist_ok=True)
        for i, segment in enumerate(segments):
            segment_path = os.path.join(SAVE_DIR, file_name, f"bass_other_seg_{i}.npy")
            np.save(segment_path, segment)
        return file_name, len(segments)
    return None


if __name__ == "__main__":
    data_dir = "/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_wav"
    track_list = os.listdir(data_dir)

    # Use multiprocessing to parallelize the processing
    num_processes = cpu_count() - 1
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_file, track_list),
            total=len(track_list),
            desc="Processing files"
        ))

    # Filter out None results and create counter
    seg_counter = {k: v for result in results if result is not None for k, v in [result]}

    print(f"Having {len(seg_counter.keys())} / {len(track_list)} usable tracks under {AMP_THRES} amplitude threshold")

    # Save the counter
    with open("../info/chorus_audio_16000_095sec_npy_bass_other_seg_counter.json", "w") as f:
        json.dump(seg_counter, f, indent=4)
