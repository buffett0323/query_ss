import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from utils import find_top_2_peak_segments  # assumed to be safe for multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

SEGMENT_TIME = 0.95
SAMPLE_RATE = 16000
AMP_THRES = 0.5
SAVE_DIR = "/mnt/gestalt/home/buffett/beatport_analyze/chorus_audio_16000_095sec_npy"
os.makedirs(SAVE_DIR, exist_ok=True)

data_dir = "/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_16000_npy"



with open("info/chorus_audio_16000_npy_track_list.txt", "r") as f:
    track_list = [line.strip() for line in f.readlines()]

def process_one(file_name, seg_time, sample_rate, amp_thres, save_dir, data_dir):
    file_path = os.path.join(data_dir, file_name, "other.npy")

    segments = find_top_2_peak_segments(file_path, seg_time, sample_rate, amp_thres)
    print(file_name, "-------", len(segments))

    if len(segments) >= 2:
        track_save_dir = os.path.join(save_dir, file_name)
        os.makedirs(track_save_dir, exist_ok=True)

        for i, segment in enumerate(segments):
            segment_path = os.path.join(track_save_dir, f"other_seg_{i}.npy")
            np.save(segment_path, segment)

        return file_name, len(segments)

    return file_name, 0


if __name__ == "__main__":
    seg_counter = {}

    with ProcessPoolExecutor() as executor:
        process_func = partial(
            process_one,
            seg_time=SEGMENT_TIME,
            sample_rate=SAMPLE_RATE,
            amp_thres=AMP_THRES,
            save_dir=SAVE_DIR,
            data_dir=data_dir
        )
        results = list(tqdm(executor.map(process_func, track_list), total=len(track_list)))

    # Aggregate results
    for file_name, count in results:
        if count >= 2:
            seg_counter[file_name] = count

    print(f"Having {len(seg_counter.keys())} / {len(track_list)} usable tracks under {AMP_THRES} amplitude threshold")

    with open("info/chorus_audio_16000_095sec_npy_seg_counter.json", "w") as f:
        json.dump(seg_counter, f, indent=4)
