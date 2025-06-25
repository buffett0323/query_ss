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


def convert_npy_to_h5(npy_dir: Path, output_h5_path: Path, splits: list = ["train", "val", "test"]):
    """
    Convert NPY files to HDF5 format for faster loading.
    
    Args:
        npy_dir: Directory containing NPY files organized by splits
        output_h5_path: Output HDF5 file path
        splits: List of data splits to convert
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
    
    args = parser.parse_args()
    
    convert_npy_to_h5(
        npy_dir=Path(args.npy_dir),
        output_h5_path=Path(args.output),
        splits=args.splits
    ) 