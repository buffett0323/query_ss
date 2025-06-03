import os
import random
import json
import soundfile as sf
import numpy as np
from shutil import copy2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def npy_to_wav(args):
    npy_file, output_path, file_name = args
    data = np.load(npy_file)
    wav_file = os.path.join(output_path, f"{file_name}.wav")
    sf.write(wav_file, data, 16000)


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
    with open("../simsiam/info/chorus_audio_16000_095sec_npy_bass_other_seg_counter.json", "r") as f:
        data = json.load(f)

    # Get the keys of the dictionary
    keys = list(data.keys())
    print(len(keys))


    # Output paths
    path = "/mnt/gestalt/home/buffett/beatport_analyze/chorus_audio_16000_095sec_npy_bass_other_new/amp_08"
    output_base = "/mnt/gestalt/home/buffett/rave/beatport_data"
    train_path = os.path.join(output_base, "train")
    test_path = os.path.join(output_base, "test")
    track = "bass_other"

    # Create output directories
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Get all files (each file represents a chorus segment)
    all_files = [os.path.join(path, f, f"{track}_seg_0.npy") for f in tqdm(keys)]
    print(len(all_files))

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
    list(tqdm(pool.imap(npy_to_wav, test_args), total=len(test_args)))


    # Copy files to train folder using multiprocessing
    print("Processing training data...")
    train_args = [(f, train_path, f.split("/")[-2]) for f in train_files]
    list(tqdm(pool.imap(npy_to_wav, train_args), total=len(train_args)))

    # Close the pool
    pool.close()
    pool.join()

    print("Done!")
