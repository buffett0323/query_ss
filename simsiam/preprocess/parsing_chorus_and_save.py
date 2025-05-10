import os
import numpy as np
import matplotlib.pyplot as plt
import json #orjson
import shutil
from tqdm import tqdm
from utils import find_top_2_peak_segments, get_top_2_peak_infos
from multiprocessing import Pool, cpu_count

SEGMENT_TIME = 0.95
SAMPLE_RATE = 16000
AMP_THRES = 0.5 # 0.5
SLICE_NAME = "amp05"
FILE_NAME = "bass_other.npy"
SEGMENT_LENGTH = int(SEGMENT_TIME * SAMPLE_RATE)
DATA_DIR = "/mnt/gestalt/home/buffett/beatport_analyze/chorus_bass_other_16000_npy"
JSON_PATH = "/mnt/gestalt/home/buffett/beatport_analyze/chorus_audio_16000_095sec_npy_bass_other_peak_info.json"
STORE_DIR = f"/mnt/gestalt/home/buffett/beatport_analyze/chorus_audio_16000_095sec_npy_bass_other_new/{SLICE_NAME}"
os.makedirs(STORE_DIR, exist_ok=True)




def process_file(file_info):
    file_name, peak_dict = file_info
    segments = []
    for key, value in peak_dict.items():
        if key != "peak_num":
            if value["amp"] >= AMP_THRES:
                start_idx = value["start_idx"]
                end_idx = start_idx + SEGMENT_LENGTH
                
                waveform = np.load(os.path.join(DATA_DIR, file_name, FILE_NAME))
                segment = waveform[start_idx:end_idx]
                segments.append(segment)
                
    
    if len(segments) >= 2:
        os.makedirs(os.path.join(STORE_DIR, file_name), exist_ok=True)
        for i, segment in enumerate(segments):
            store_path = os.path.join(STORE_DIR, file_name, f"bass_other_seg_{i}.npy")
            np.save(store_path, segment)
    
    return file_name, len(segments)


if __name__ == "__main__":
    with open(JSON_PATH, "r") as f:
        peak_info_dict = json.load(f)

    # Process files in parallel
    num_processes = cpu_count() - 1
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_file, peak_info_dict.items()),
            total=len(peak_info_dict),
            desc="Processing files"
        ))
    
    # Combine results into seg_counter dictionary
    seg_counter = dict()
    for file_name, num_segments in results:
        if num_segments >= 2:
            seg_counter[file_name] = num_segments
    
    with open(f"../../moco/info/train_seg_counter_{SLICE_NAME}.json", "w") as f:
        json.dump(seg_counter, f, indent=4)
