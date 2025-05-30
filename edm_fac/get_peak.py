import os
import librosa
import numpy as np
import scipy.signal
from tqdm import tqdm
import json

AMP_THRES = 0.5
SAMPLE_RATE = 44100

path = "/mnt/gestalt/home/buffett/EDM_FAC_DATA/rendered_audio"
peak_records = {}

for stem in ["lead", "pad", "bass", "keys", "pluck"]:

    list_files = os.listdir(os.path.join(path, stem))
    for file in tqdm(list_files):
        if file.endswith(".wav"):
            audio, sr = librosa.load(os.path.join(path, stem, file), sr=None)
            
            # Find indices where amplitude exceeds threshold
            envelope = np.abs(audio)
            peaks, _ = scipy.signal.find_peaks(envelope, distance=SAMPLE_RATE // 20)

            peaks_in_seconds = [peak / SAMPLE_RATE for peak in peaks]

            # Store the peak times in seconds
            peak_records[file.split(".wav")[0]] = peaks_in_seconds


with open("/mnt/gestalt/home/buffett/EDM_FAC_DATA/peak_records.json", "w") as f:
    json.dump(peak_records, f)