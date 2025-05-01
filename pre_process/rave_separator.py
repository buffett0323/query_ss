import os
import random
from shutil import copy2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def copy_file(args):
    src_file, dst_path = args
    song_name = src_file.split("/")[-2]
    inst_name = os.path.basename(src_file)
    dst_file = os.path.join(dst_path, f"{song_name}_{inst_name}")
    copy2(src_file, dst_file)


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Output paths
    path = "/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_wav"
    output_base = "/mnt/gestalt/home/buffett/rave/beatport_data"
    train_path = os.path.join(output_base, "train")
    test_path = os.path.join(output_base, "test")
    track = "bass_other"

    # Create output directories
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Get all files (each file represents a chorus segment)
    all_files = [os.path.join(path, f, f"{track}.wav") for f in tqdm(os.listdir(path))]
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

    # Copy files to train folder using multiprocessing
    print("Processing training data...")
    train_args = [(f, train_path) for f in train_files]
    list(tqdm(pool.imap(copy_file, train_args), total=len(train_args)))

    # Copy files to test folder using multiprocessing
    print("Processing test data...")
    test_args = [(f, test_path) for f in test_files]
    list(tqdm(pool.imap(copy_file, test_args), total=len(test_args)))

    # Close the pool
    pool.close()
    pool.join()

    print("Done!")
