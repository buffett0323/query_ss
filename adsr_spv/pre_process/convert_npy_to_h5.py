#!/usr/bin/env python3
"""
Convert NPY files to HDF5 format for faster data loading.
This script will significantly speed up training by eliminating individual file I/O.
"""

import json
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import multiprocessing as mp


def load_single_file(args):
    """Load a single NPY file and its ADSR parameters."""
    npy_dir, split, chunk = args
    mel_path = npy_dir / split / chunk["npy_file"]
    mel = np.load(mel_path)
    adsr = np.array([chunk["attack"], chunk["decay"], chunk["sustain"], chunk["release"]], dtype=np.float32)
    return mel, adsr


def load_chunk_parallel(npy_dir, split, metadata_chunk, num_workers=None):
    """Load a chunk of files in parallel."""
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(metadata_chunk))

    # Prepare arguments for multiprocessing
    args_list = [(npy_dir, split, chunk) for chunk in metadata_chunk]

    # Use multiprocessing to load files in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(load_single_file, args_list),
            total=len(args_list),
            desc=f"Loading {split} data (parallel)"
        ))

    # Separate mel and adsr data
    mel_list, adsr_list = zip(*results)
    return list(mel_list), list(adsr_list)


def convert_npy_to_h5(npy_dir: Path, output_h5_path: Path, splits: list = ["train", "val", "test"], num_workers: int = None):
    """
    Convert NPY files to HDF5 format for faster loading.

    Args:
        npy_dir: Directory containing NPY files organized by splits
        output_h5_path: Output HDF5 file path
        splits: List of data splits to convert
        num_workers: Number of worker processes (None = auto-detect)
    """

    if num_workers is None:
        num_workers = min(mp.cpu_count(), 24)

    print(f"Using {num_workers} worker processes for parallel loading")

    with h5py.File(output_h5_path, 'w') as h5_file:
        for split in splits:
            print(f"\nConverting {split} split...")

            # Load metadata
            metadata_path = npy_dir / split / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            print(f"Found {len(metadata)} files to convert")

            # Load all NPY files in parallel
            mel_list, adsr_list = load_chunk_parallel(npy_dir, split, metadata, num_workers)

            # Convert to numpy arrays
            print(f"Stacking {split} data...")
            mel_array = np.stack(mel_list, axis=0)
            adsr_array = np.stack(adsr_list, axis=0)

            # Save to HDF5 with compression
            print(f"Saving {split} data to HDF5...")
            h5_file.create_dataset(f'{split}_mel', data=mel_array, compression='gzip', compression_opts=6)
            h5_file.create_dataset(f'{split}_adsr', data=adsr_array, compression='gzip', compression_opts=6)

            # Store metadata
            h5_file.attrs[f'{split}_num_samples'] = len(metadata)
            h5_file.attrs[f'{split}_mel_shape'] = mel_array.shape
            h5_file.attrs[f'{split}_adsr_shape'] = adsr_array.shape

            print(f"{split}: {len(metadata)} samples, mel shape: {mel_array.shape}, adsr shape: {adsr_array.shape}")

    print(f"\nConversion complete! Saved to: {output_h5_path}")
    print(f"File size: {output_h5_path.stat().st_size / (1024**3):.2f} GB")


def convert_npy_to_h5_sequential(npy_dir: Path, output_h5_path: Path, splits: list = ["train", "val", "test"]):
    """
    Sequential version for comparison or when multiprocessing is not desired.
    """

    with h5py.File(output_h5_path, 'w') as h5_file:
        for split in splits:
            print(f"Converting {split} split...")

            # Load metadata
            metadata_path = npy_dir / split / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Prepare arrays
            mel_list, adsr_list = [], []

            # Load all NPY files
            for chunk in tqdm(metadata, desc=f"Loading {split} data"):
                mel_path = npy_dir / split / chunk["npy_file"]
                mel = np.load(mel_path)
                mel_list.append(mel)

                adsr = np.array([chunk["attack"], chunk["decay"], chunk["sustain"], chunk["release"]], dtype=np.float32)
                adsr_list.append(adsr)

            # Convert to numpy arrays
            mel_array = np.stack(mel_list, axis=0)
            adsr_array = np.stack(adsr_list, axis=0)

            # Save to HDF5
            h5_file.create_dataset(f'{split}_mel', data=mel_array, compression='gzip', compression_opts=6)
            h5_file.create_dataset(f'{split}_adsr', data=adsr_array, compression='gzip', compression_opts=6)

            # Store metadata
            h5_file.attrs[f'{split}_num_samples'] = len(metadata)
            h5_file.attrs[f'{split}_mel_shape'] = mel_array.shape
            h5_file.attrs[f'{split}_adsr_shape'] = adsr_array.shape

            print(f"{split}: {len(metadata)} samples, mel shape: {mel_array.shape}, adsr shape: {adsr_array.shape}")

    print(f"Conversion complete! Saved to: {output_h5_path}")
    print(f"File size: {output_h5_path.stat().st_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NPY files to HDF5 format")
    parser.add_argument(
        "--npy_dir",
        type=str,
        default="/home/buffett/dataset/rendered_adsr_unpaired_mel_npy",
        help="Directory containing NPY files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/buffett/dataset/rendered_adsr_unpaired_mel_h5/adsr_mel.h5",
        help="Output HDF5 file path"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Data splits to convert"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=24,
        help="Number of worker processes (None = auto-detect)"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential processing instead of multiprocessing"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.sequential:
        print("Using sequential processing...")
        convert_npy_to_h5_sequential(
            npy_dir=Path(args.npy_dir),
            output_h5_path=output_path,
            splits=args.splits
        )
    else:
        print("Using multiprocessing...")
        convert_npy_to_h5(
            npy_dir=Path(args.npy_dir),
            output_h5_path=output_path,
            splits=args.splits,
            num_workers=args.num_workers
        )
