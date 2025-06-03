import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from utils import find_top_2_peak_segments, plot_waveform, npy2audio

SEGMENT_TIME = 0.95
SAMPLE_RATE = 16000
AMP_THRES = 0.5
SAVE_DIR = "/mnt/gestalt/home/buffett/beatport_analyze/chorus_audio_16000_095sec_npy"
os.makedirs(SAVE_DIR, exist_ok=True)

if __name__ == "__main__":
    data_dir = "/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_16000_npy"

    with open("info/chorus_audio_16000_npy_track_list.txt", "r") as f:
        track_list = [line.strip() for line in f.readlines()]

    seg_counter = dict()

    for file_name in tqdm(track_list):
        file_path = os.path.join(data_dir, file_name, "other.npy")

        # # 1. Plot the waveform
        # plot_waveform(file_path)


        # 2. Find the top 2 peaks
        segments = find_top_2_peak_segments(file_path, SEGMENT_TIME, SAMPLE_RATE, AMP_THRES)
        # print(len(segments), segments[0].shape, segments[1].shape)


        # 3. Listen to audio to test peak position
        # npy2audio(file_path)


        # 4. Save the segments & Update the counter
        print(file_name, "-------", len(segments))

        if len(segments) >= 2:
            os.makedirs(os.path.join(SAVE_DIR, file_name), exist_ok=True)
            for i, segment in enumerate(segments):
                segment_path = os.path.join(SAVE_DIR, file_name, f"other_seg_{i}.npy")
                np.save(segment_path, segment)

            seg_counter[file_name] = len(segments)


    print(f"Having {len(seg_counter.keys())} / {len(track_list)} usable tracks under {AMP_THRES} amplitude threshold")


    # 5. Save the counter
    with open("../info/chorus_audio_16000_095sec_npy_seg_counter.json", "w") as f:
        json.dump(seg_counter, f, indent=4)
