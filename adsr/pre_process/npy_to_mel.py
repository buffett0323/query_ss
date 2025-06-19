import os
import numpy as np
import librosa
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def process_single_file(file, input_path, output_path):
    try:
        if file.endswith(".npy"):
            wav = np.load(os.path.join(input_path, file))
            mel = librosa.feature.melspectrogram(
                y=wav,
                sr=44100,
                n_mels=128,
                fmin=20,
                fmax=22050,
                hop_length=512,
                n_fft=2048,
                window='hann',
                center=True,
                power=2.0,
                htk=True,  # Use HTK formula for mel scale (matches torchaudio)
                norm='slaney'  # Use Slaney normalization (matches torchaudio)
            )
            np.save(os.path.join(output_path, file.replace(".npy", ".npy")), mel)
            return True
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        return False

def process_chunk(chunk_files, input_path, output_path):
    """Process a chunk of files"""
    results = []
    for file in chunk_files:
        result = process_single_file(file, input_path, output_path)
        results.append(result)
    return results

if __name__ == "__main__":
    input_path = "/mnt/gestalt/home/buffett/rendered_adsr_dataset_npy"
    output_path = "/mnt/gestalt/home/buffett/rendered_adsr_dataset_npy_mel"

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Get all files
    files = [f for f in os.listdir(input_path) if f.endswith(".npy")]

    # Set up multiprocessing
    num_workers = mp.cpu_count()
    chunk_size = 100  # Process 100 files per chunk

    # Split files into chunks
    chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    print(f"Split {len(files)} files into {len(chunks)} chunks")

    # Create process pool
    pool = mp.Pool(num_workers)

    # Process chunks in parallel
    all_results = []
    for chunk_results in tqdm(
        pool.imap(partial(process_chunk, input_path=input_path, output_path=output_path), chunks),
        total=len(chunks),
        desc="Processing chunks"
    ):
        all_results.extend(chunk_results)

    pool.close()
    pool.join()

    # Print summary
    successful = sum(all_results)
    print(f"Successfully processed {successful} files")
    print(f"Failed to process {len(files) - successful} files")
