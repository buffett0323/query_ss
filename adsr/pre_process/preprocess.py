import os
import json
import h5py
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
N_MELS = 128

def process_single_file(path, folder):
    try:
        # Load audio
        mel_spec = np.load(os.path.join(folder, path))

        # Add channel dimension if it doesn't exist
        if mel_spec.ndim == 2:
            mel_spec = mel_spec[np.newaxis, :, :]  # Add channel dimension: (128, 256) -> (1, 128, 256)

        # Debug: print shape for first few files
        if path in ["0_0_mel.npy", "0_1_mel.npy", "0_2_mel.npy"]:
            print(f"Debug - {path} shape: {mel_spec.shape}")

        # Get environment ID
        env_id = int(path.split("_")[1])

        return {
            'mel_spec': mel_spec,
            'env_id': env_id,
            'path': path
        }
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        return None

def process_chunk(chunk_paths, folder):
    """Process a chunk of files and return results"""
    results = []
    for path in chunk_paths:
        result = process_single_file(path, folder)
        if result is not None:
            results.append(result)
    return results

def preprocess_and_save(folder, output_path, num_workers=None, chunk_size=1000, json_path=None):
    if num_workers is None:
        num_workers = mp.cpu_count()

    print(f"Using {num_workers} workers for preprocessing")

    # Load metadata
    with open(json_path, "r") as f:
        metadata = json.load(f)

    # Get all file paths
    paths = [chunk["file"].replace(".wav", "_mel.npy") for chunk in metadata]#[:1800]

    # Split paths into chunks
    chunks = [paths[i:i + chunk_size] for i in range(0, len(paths), chunk_size)]
    print(f"Split {len(paths)} files into {len(chunks)} chunks")

    # Create process pool
    pool = mp.Pool(num_workers)

    # Process chunks in parallel
    all_results = []
    for chunk_results in tqdm(
        pool.imap(partial(process_chunk, folder=folder), chunks),
        total=len(chunks),
        desc="Processing chunks"
    ):
        all_results.extend(chunk_results)

    pool.close()
    pool.join()

    # Sort results by path to maintain consistent order
    all_results.sort(key=lambda x: x['path'])
    print(f"Total processed files: {len(all_results)}")

    # Calculate appropriate chunk sizes
    mel_chunk_size = min(50, len(all_results))  # Smaller chunks for mel spectrograms
    data_chunk_size = min(100, len(all_results))  # Larger chunks for other data

    # Create h5 file with chunked storage
    print(f"Creating h5 file at {output_path}")
    with h5py.File(output_path, 'w') as f:
        # Create datasets with chunked storage and compression
        mel_specs = f.create_dataset(
            'mel_specs',
            shape=(len(all_results), 1, N_MELS, 256),
            dtype='float32',
            compression='gzip',
            compression_opts=9,  # Maximum compression
            chunks=(mel_chunk_size, 1, N_MELS, 256),  # Adjusted chunk size
            shuffle=True  # Enable shuffle filter for better compression
        )

        env_ids = f.create_dataset(
            'env_ids',
            shape=(len(all_results),),
            dtype='int32',
            compression='gzip',
            compression_opts=9,
            chunks=(data_chunk_size,),  # Adjusted chunk size
            shuffle=True
        )

        file_paths = f.create_dataset(
            'file_paths',
            shape=(len(all_results),),
            dtype=h5py.special_dtype(vlen=str),
            compression='gzip',
            compression_opts=9,
            chunks=(data_chunk_size,),  # Adjusted chunk size
            shuffle=True
        )

        # Store results in chunks
        print("Finished processing, storing results in chunks")
        write_chunk_size = min(500, len(all_results))  # Increased chunk size for better performance
        total_chunks = (len(all_results) + write_chunk_size - 1) // write_chunk_size

        for i in tqdm(range(0, len(all_results), write_chunk_size),
                     total=total_chunks,
                     desc="Storing results"):
            chunk = all_results[i:i + write_chunk_size]
            # Prepare batch data
            batch_mel_specs = np.stack([result['mel_spec'] for result in chunk])
            batch_env_ids = np.array([result['env_id'] for result in chunk])
            batch_paths = np.array([result['path'] for result in chunk], dtype=object)

            # Validate shapes before writing
            expected_shape = (len(chunk), 1, N_MELS, 256)
            if batch_mel_specs.shape != expected_shape:
                print(f"Shape mismatch! Expected {expected_shape}, got {batch_mel_specs.shape}")
                print(f"First mel_spec shape: {chunk[0]['mel_spec'].shape}")
                raise ValueError(f"Mel spectrogram shape mismatch: {batch_mel_specs.shape} vs {expected_shape}")

            # Write batch at once
            mel_specs[i:i + len(chunk)] = batch_mel_specs
            env_ids[i:i + len(chunk)] = batch_env_ids
            file_paths[i:i + len(chunk)] = batch_paths

    print(f"Successfully processed {len(all_results)} files")
    print(f"Failed to process {len(paths) - len(all_results)} files")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess audio files to h5 format')
    parser.add_argument('--input_folder', type=str,
                        default="/mnt/gestalt/home/buffett/rendered_adsr_dataset_npy_new_mel",
                       help='path to folder containing audio files and metadata.json')
    parser.add_argument('--output_path', type=str,
                        default="/mnt/gestalt/home/buffett/adsr_h5/adsr_new_mel.h5",
                       help='path to save the h5 file')
    parser.add_argument('--num_workers', type=int, default=24,
                       help='number of worker processes (default: number of CPU cores)')
    parser.add_argument('--chunk_size', type=int, default=500,
                       help='number of files to process in each chunk')
    parser.add_argument('--json_path', type=str,
                        default="/mnt/gestalt/home/buffett/rendered_adsr_dataset/metadata.json",
                       help='path to metadata.json')
    args = parser.parse_args()

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    preprocess_and_save(
        args.input_folder,
        args.output_path,
        args.num_workers,
        args.chunk_size,
        args.json_path
    )
