import numpy as np
import scipy.signal
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Directory Paths
seconds = 8
input_dir = "/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_16000_npy"  
output_dir = f"/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_16000_{seconds}secs_npy"  
sources = ["other.npy"] #["bass.npy", "drums.npy", "mix.npy", "other.npy", "vocals.npy"]

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)
slice_duration = 16000 * seconds  # 6 seconds

# # Resampling function
# def resample_audio(audio, orig_sr=44100, target_sr=16000):
#     """Efficiently resample audio using polyphase filtering."""
#     audio = np.mean(audio, axis=0, keepdims=False)  # Convert to mono
#     return scipy.signal.resample_poly(audio, target_sr, orig_sr)



# Function to process a single file
def process_file(filename):
    input_subdir = os.path.join(input_dir, filename)
    output_subdir = os.path.join(output_dir, filename)

    os.makedirs(output_subdir, exist_ok=True)  # Ensure subdirectory exists

    for source in sources:
        filepath = os.path.join(input_subdir, source)
        output_path = os.path.join(output_subdir, source)

        if not os.path.exists(filepath):  # Skip if file doesn't exist
            continue
        
        # Load .npy file
        audio_data = np.load(filepath)  # Shape: (samples, ) or (samples, channels)
        if audio_data.shape[0] < slice_duration:
            continue
        audio_data = audio_data[:slice_duration]

        # Save Resampled Data
        np.save(output_path, audio_data)

    return f"Processed {filename}"



# Get all filenames
file_list = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]



# Use multiprocessing to process files in parallel
if __name__ == "__main__":
    num_workers = cpu_count()-2

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(process_file, file_list), total=len(file_list)))

    print("Resampling complete!")
