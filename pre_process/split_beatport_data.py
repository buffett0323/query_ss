#!/usr/bin/env python3
"""
Script to split beatport data into train/evaluation sets with 95:5 ratio
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import argparse

def get_audio_files(directory):
    """Get all audio files from directory"""
    audio_extensions = {'.wav', '.mp3', '.flac', '.aiff', '.m4a', '.ogg'}
    audio_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                audio_files.append(os.path.join(root, file))

    return audio_files

def split_data(source_dir, train_ratio=0.95, seed=42, move_files=False):
    """
    Split data into train/evaluation sets

    Args:
        source_dir: Path to source directory containing audio files
        train_ratio: Ratio for training data (default 0.95 for 95%)
        seed: Random seed for reproducibility
        move_files: If True, move files instead of copying them
    """

    source_path = Path(source_dir)
    if not source_path.exists():
        raise ValueError(f"Source directory {source_dir} does not exist!")

    # Create train and evaluation directories
    train_dir = source_path / "train"
    val_dir = source_path / "evaluation"

    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    print(f"Created directories:")
    print(f"  Train: {train_dir}")
    print(f"  Evaluation: {val_dir}")

    # Get all audio files
    print("Scanning for audio files...")
    audio_files = get_audio_files(source_dir)

    # Filter out files that are already in train/val directories
    audio_files = [f for f in audio_files if not (str(train_dir) in f or str(val_dir) in f)]

    if not audio_files:
        print("No audio files found to split!")
        return

    print(f"Found {len(audio_files)} audio files")

    # Shuffle files with seed for reproducibility
    random.seed(seed)
    random.shuffle(audio_files)

    # Calculate split point
    split_point = int(len(audio_files) * train_ratio)
    train_files = audio_files[:split_point]
    val_files = audio_files[split_point:]

    print(f"Splitting data:")
    print(f"  Train: {len(train_files)} files ({len(train_files)/len(audio_files)*100:.1f}%)")
    print(f"  Evaluation: {len(val_files)} files ({len(val_files)/len(audio_files)*100:.1f}%)")

    # Copy or move files to train directory
    action = "Moving" if move_files else "Copying"
    print(f"\n{action} files to train directory...")
    for file_path in tqdm(train_files, desc="Train files"):
        src = Path(file_path)
        # Maintain directory structure if files are in subdirectories
        rel_path = src.relative_to(source_path)

        # Skip if already in train/val dirs
        if str(rel_path).startswith(('train/', 'evaluation/')):
            continue

        dst = train_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            if move_files:
                shutil.move(src, dst)
            else:
                shutil.copy2(src, dst)
        except Exception as e:
            action_verb = "moving" if move_files else "copying"
            print(f"Error {action_verb} {src}: {e}")

    # Copy or move files to evaluation directory
    print(f"{action} files to evaluation directory...")
    for file_path in tqdm(val_files, desc="Evaluation files"):
        src = Path(file_path)
        rel_path = src.relative_to(source_path)

        # Skip if already in train/val dirs
        if str(rel_path).startswith(('train/', 'evaluation/')):
            continue

        dst = val_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            if move_files:
                shutil.move(src, dst)
            else:
                shutil.copy2(src, dst)
        except Exception as e:
            action_verb = "moving" if move_files else "copying"
            print(f"Error {action_verb} {src}: {e}")

    print("\nData splitting completed!")
    print(f"Train files: {len(list(train_dir.rglob('*')))} files")
    print(f"Evaluation files: {len(list(val_dir.rglob('*')))} files")

def main():
    parser = argparse.ArgumentParser(description="Split beatport data into train/evaluation sets")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="/home/buffett/dataset/EDM_FAC_DATA/beatport",
        help="Source directory containing audio files"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.95,
        help="Ratio for training data (default: 0.95 for 95%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without actually copying/moving files"
    )

    args = parser.parse_args()

    print("Beatport Data Splitter")
    print("=" * 50)
    print(f"Source directory: {args.source_dir}")
    print(f"Train ratio: {args.train_ratio} ({args.train_ratio*100}%)")
    print(f"Evaluation ratio: {1-args.train_ratio} ({(1-args.train_ratio)*100}%)")
    print(f"Random seed: {args.seed}")
    print(f"Operation: {'Move' if args.move else 'Copy'} files")

    if args.dry_run:
        action_text = "moved" if args.move else "copied"
        print(f"\n*** DRY RUN MODE - No files will be {action_text} ***")

        source_path = Path(args.source_dir)
        if not source_path.exists():
            print(f"Error: Source directory {args.source_dir} does not exist!")
            return

        audio_files = get_audio_files(args.source_dir)
        audio_files = [f for f in audio_files if not ('train/' in f or 'evaluation/' in f)]

        if not audio_files:
            print("No audio files found to split!")
            return

        print(f"Found {len(audio_files)} audio files")

        split_point = int(len(audio_files) * args.train_ratio)
        train_count = split_point
        val_count = len(audio_files) - split_point

        print(f"Would create:")
        print(f"  Train: {train_count} files ({train_count/len(audio_files)*100:.1f}%)")
        print(f"  Evaluation: {val_count} files ({val_count/len(audio_files)*100:.1f}%)")

    else:
        try:
            split_data(args.source_dir, args.train_ratio, args.seed, args.move)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
