import os
import librosa
import numpy as np
import scipy.signal
from tqdm import tqdm
import json
from multiprocessing import Pool, Manager
from functools import partial

def process_file(file_info, path, peak_records):
    stem, file = file_info
    if file.endswith(".wav"):
        audio, sr = librosa.load(os.path.join(path, stem, file), sr=None)
        
        # Find indices where amplitude exceeds threshold
        envelope = np.abs(audio)
        peaks, _ = scipy.signal.find_peaks(envelope, distance=SAMPLE_RATE // 20)

        peaks_in_seconds = [peak / SAMPLE_RATE for peak in peaks]

        # Store the peak times in seconds
        peak_records[file.split(".wav")[0]] = peaks_in_seconds

AMP_THRES = 0.5
SAMPLE_RATE = 44100

path = "/mnt/gestalt/home/buffett/EDM_FAC_DATA/rendered_audio"

# Create a manager to handle shared dictionary between processes
manager = Manager()
peak_records = manager.dict()

# Create list of all files to process
files_to_process = []
for stem in ["lead", "pad", "bass", "keys", "pluck"]:
    list_files = os.listdir(os.path.join(path, stem))
    files_to_process.extend([(stem, file) for file in list_files])

# Create pool of workers and process files
with Pool() as pool:
    process_func = partial(process_file, path=path, peak_records=peak_records)
    list(tqdm(pool.imap(process_func, files_to_process), total=len(files_to_process)))

# Convert manager dict to regular dict for JSON serialization
peak_records_dict = dict(peak_records)

with open("/mnt/gestalt/home/buffett/EDM_FAC_DATA/peak_records.json", "w") as f:
    json.dump(peak_records_dict, f)