import os
import torchaudio
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

# Define dataset split
split = "train"
base = "/home/buffett/NAS_NTU"
coco_path = f"{base}/cocochorales_output/main_dataset/{split}"

# Get all WAV files
def get_wav_files():
    file_list = []
    for track in tqdm(os.listdir(coco_path)):
        track_path = os.path.join(coco_path, track, "stems_audio")
        if os.path.exists(track_path):  # Ensure folder exists
            for stem in os.listdir(track_path):
                if stem.endswith(".wav"):
                    file_list.append(os.path.join(track_path, stem))
    return file_list

# Convert WAV to NPY
def process_wav(wav_path):
    try:
        wav, _ = torchaudio.load(wav_path)  # Load audio
        npy_path = wav_path.replace(".wav", ".npy")
        np.save(npy_path, wav.numpy())  # Save as .npy
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")

# Multiprocessing Execution
def main():
    wav_files = get_wav_files()
    print(f"Found {len(wav_files)} WAV files. Processing...")

    # Use multiprocessing Pool
    # num_workers = max(mp.cpu_count()-2, 24)
    
    # with mp.Pool(num_workers) as pool:
    #     list(tqdm(pool.imap(process_wav, wav_files), total=len(wav_files)))

    print("✅ All WAV files converted to NPY!")

if __name__ == "__main__":
    main()
