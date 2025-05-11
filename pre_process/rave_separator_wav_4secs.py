import os
import random
import json
import soundfile as sf
import numpy as np
from shutil import copy2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def parse_4_secs(args):
    wav_file, output_path, track_name = args
    data, sr = sf.read(wav_file)
    
    # Parse 4 seconds
    if data.shape[0] > 44100 * 4:
        data = data[:44100 * 4]
    else:
        print("data.shape:", data.shape, "sr:", sr)
    
    # Convert to mono
    data = data.mean(axis=1)
    
    # Output path
    write_path = os.path.join(output_path, f"{track_name}.wav")
    sf.write(write_path, data, sr)
    
    

# def npy_to_wav(args):
#     npy_file, output_path, file_name = args
#     data = np.load(npy_file)
#     wav_file = os.path.join(output_path, f"{file_name}.wav")
#     sf.write(wav_file, data, 16000)


# def copy_file(args):
#     src_file, dst_path = args
#     song_name = src_file.split("/")[-2]
#     inst_name = os.path.basename(src_file)
#     dst_file = os.path.join(dst_path, f"{song_name}_{inst_name}")
#     copy2(src_file, dst_file)


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Load json file
    with open("../moco/info/train_seg_counter_amp_05.json", "r") as f:
        data = json.load(f)
    
    # Get the keys of the dictionary
    keys = list(data.keys())
    print("len(keys):", len(keys))
    

    # Output paths
    path = "/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_wav"
    output_base = "/mnt/gestalt/home/buffett/rave/beatport_data_4secs_mono"
    train_path = os.path.join(output_base, "train")
    test_path = os.path.join(output_base, "test")
    track = "bass_other"

    # Create output directories
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Get all files (each file represents a chorus segment)
    all_files = [os.path.join(path, f, f"{track}.wav") for f in tqdm(keys)]
    print("all wav files length:", len(all_files))

    
    # Shuffle and split into train/test (90/10 split)
    random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.9)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]
    print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")

    # Set up multiprocessing pool
    num_processes = cpu_count() - 1
    pool = Pool(processes=num_processes)


    # Copy files to test folder using multiprocessing
    print("Processing test data...")
    test_args = [(f, test_path, f.split("/")[-2]) for f in test_files]
    list(tqdm(pool.imap(parse_4_secs, test_args), total=len(test_args)))
    
    
    # Copy files to train folder using multiprocessing
    print("Processing training data...")
    train_args = [(f, train_path, f.split("/")[-2]) for f in train_files]
    list(tqdm(pool.imap(parse_4_secs, train_args), total=len(train_args)))

    # Close the pool
    pool.close()
    pool.join()

    print("Done!")
