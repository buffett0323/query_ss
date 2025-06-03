import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from utils import find_top_2_peak_segments, get_top_2_peak_infos
from multiprocessing import Pool, cpu_count

SEGMENT_TIME = 0.95
SAMPLE_RATE = 16000
AMP_THRES = 0.5 # 0.5
FILE_NAME = "bass_other.npy"


def process_file(file_name):
    file_path = os.path.join(data_dir, file_name, FILE_NAME)
    result = get_top_2_peak_infos(
        file_path,
        segment_duration=SEGMENT_TIME,
        sample_rate=SAMPLE_RATE,
        amp_thres=AMP_THRES
    )


    if not result:
        return file_name, {}


    result_dict = {}
    result_dict["peak_num"] = len(result)
    for i, (peak_idx, amp, time_sec) in enumerate(result):
        result_dict[f"{file_name}_{i}"] =  {
            "start_idx": int(peak_idx),
            "amp": float(amp),
            "start_sec": float(time_sec)
        }
        # [int(peak_idx), float(amp), float(time_sec)]

    return file_name, result_dict



if __name__ == "__main__":
    data_dir = "/mnt/gestalt/home/buffett/beatport_analyze/chorus_bass_other_16000_npy"

    with open("../info/chorus_audio_16000_npy_track_list.txt", "r") as f:
        track_list = [line.strip() for line in f.readlines()]

    # Use multiprocessing to parallelize the processing
    num_processes = cpu_count() - 1
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_file, track_list),
            total=len(track_list),
            desc="Processing files"
        ))

    # Combine all peak info dictionaries
    peak_info_dict = {}
    for (file_name, peak_dict) in tqdm(results, desc="Write peak info into dict"):
        if peak_dict:
            peak_info_dict[file_name] = peak_dict

    print(f"Having {len(peak_info_dict.keys())} peaks found under {AMP_THRES} amplitude threshold")

    # Save the peak info dictionary
    with open("../info/chorus_audio_16000_095sec_npy_bass_other_peak_info.json", "w") as f:
        json.dump(peak_info_dict, f, indent=4)
