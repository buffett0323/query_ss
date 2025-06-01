import os
import librosa
import numpy as np
import scipy.signal
from tqdm import tqdm
import json
from multiprocessing import Pool, cpu_count
from functools import partial


SAMPLE_RATE = 44100
STEMS = ['lead'] #['lead', 'pad', 'bass', 'keys', 'pluck']
FILTER_TIME = 8
path = "/mnt/gestalt/home/buffett/EDM_FAC_DATA/rendered_audio_new"


def process_file(file_info, path):
    stem, file = file_info
    file_name = file.split(".wav")[0]
    if file.endswith(".wav"):
        audio, _ = librosa.load(os.path.join(path, stem, file), sr=None)
        audio = audio[:int(FILTER_TIME * SAMPLE_RATE)]
        
        # Find indices where amplitude exceeds threshold
        envelope = np.abs(audio)
        peaks, _ = scipy.signal.find_peaks(envelope, distance=SAMPLE_RATE // 20)
        if len(peaks) == 0: return (file_name, [])
        
        # Get peak amplitudes
        peak_amplitudes = envelope[peaks]
        
        # Get top 10 peaks by amplitude
        if len(peaks) > 10:
            top_indices = np.argsort(peak_amplitudes)[-10:]
            peaks = peaks[top_indices]
            peak_amplitudes = peak_amplitudes[top_indices]
        
        # Convert peak indices to time and filter peaks > 8 seconds
        peak_info = [peak_idx / SAMPLE_RATE for peak_idx in peaks]
        
        return (file_name, peak_info) if peak_info else (file_name, [])


if __name__ == "__main__":
    # Create list of all files to process
    files_to_process = []
    for stem in STEMS:
        list_files = os.listdir(os.path.join(path, stem))
        files_to_process.extend([(stem, file) for file in list_files])
        
    # files_to_process = files_to_process[:10]
    print(len(files_to_process))

    # Process files in parallel using 24 processes
    process_with_path = partial(process_file, path=path)
    num_processes = cpu_count() - 1
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_with_path, files_to_process),
            total=len(files_to_process),
            desc="Processing files"
        ))
    
    # Filter out None results and separate files with no peaks
    no_peak_files = []
    peak_records = {}
    for result in results:
        if len(result[1]) == 0:
            no_peak_files.append(result[0])
        else:
            file_name, peak_info = result
            peak_records[file_name] = peak_info
            
    print(f"Files with no peaks: {len(no_peak_files)}")
    if no_peak_files:
        with open("info/no_peak_files.txt", "w") as f:
            for file in no_peak_files:
                f.write(file + "\n")
    
    
    # Save results
    with open("/mnt/gestalt/home/buffett/EDM_FAC_DATA/peak_records_lead.json", "w") as f:
        json.dump(peak_records, f, indent=4)