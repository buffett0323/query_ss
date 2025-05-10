# Using original track, without .dbfss
import os
import random
import math
import pickle
import torch
import torchaudio
import json
import librosa
import librosa.display
import argparse
import scipy.interpolate
import scipy.stats
import nnAudio.features
import lmdb
import pickle


import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torchaudio.transforms as T
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchaudio.transforms import MelSpectrogram
from pytorch_lightning import LightningDataModule
from typing import Optional
from tqdm import tqdm
from scipy.stats import entropy

from audiomentations import Compose, TimeStretch, PitchShift, TimeMaskBack, SeqPerturb_Reverse
from utils import yaml_config_hook
# import audiomentations
# print("Audiomentations Loaded in Dataset.py:", audiomentations.__file__)



def create_lmdb(data_dir, lmdb_path):
    map_size = 10 * 1024 ** 3
    env = lmdb.open(lmdb_path, map_size=map_size)
    
    with open(f"info/chorus_audio_16000_095sec_npy_bass_other_seg_counter.json", "r") as f:
        seg_counter = json.load(f)
    
    # Pre-collect all file paths to avoid nested loops
    npy_files = []
    for song, _ in tqdm(seg_counter.items(), desc="Processing songs"):
        npy_files.append(
            (song, os.path.join(data_dir, song, "bass_other_seg_0.npy")) 
        )
    print("len(npy_files):", len(npy_files))
    
    # Batch write to LMDB    
    batch_size = 1000
    txn = env.begin(write=True)

    for i, (song, file_path) in enumerate(tqdm(npy_files, desc="Writing to LMDB in batch")):
        arr = np.load(file_path)
        key = f"{song}/{os.path.basename(file_path)}".encode()
        txn.put(key, pickle.dumps(arr, protocol=4))
        
        if (i + 1) % batch_size == 0:
            txn.commit()
            txn = env.begin(write=True)

    # Final commit
    txn.commit()
    print(f"LMDB created at {lmdb_path}")


# Beatport Dataset
class SegmentBPDataset(Dataset):
    """ For 4 seconds audio data """
    def __init__(
        self,
        data_dir,
        split="train",
        stem="bass_other", #["vocals", "bass", "drums", "other"], # VBDO
        eval_id=0,
        eval_mode=False,
        sample_rate=16000,
        sp_method="fixed", # "random", "fixed"
        num_seq_segments=5,
        fixed_second=0.3,
        p_ts=0.5,
        p_ps=0.5, # 0.4
        p_tm=0.5,
        p_tstr=0.5,
        semitone_range=[-2, 2], #[-4, 4],
        tm_min_band_part=0.05, #0.1,
        tm_max_band_part=0.1, #0.15,
        tm_fade=True,
        tstr_min_rate=0.8,
        tstr_max_rate=1.25,
        use_lmdb=False, # True
        amp_name="_amp08",
    ):
        # Load segment info list

        with open(f"info/{split}_seg_counter{amp_name}.json", "r") as f:
            self.seg_counter = json.load(f)
            self.bp_listdir = list(self.seg_counter.keys())
            print(f"{split} Mode: {len(self.bp_listdir)} songs")

        self.label_dict = {song: i for i, song in enumerate(self.bp_listdir)}
        self.data_dir = data_dir
        self.split = split
        self.stem = stem
        self.eval_mode = eval_mode
        self.eval_id = eval_id
        self.sample_rate = sample_rate
        self.use_lmdb = use_lmdb
        
        # LMDB
        if self.use_lmdb:
            self.lmdb_env = lmdb.open(data_dir, readonly=True, lock=False, readahead=False, meminit=False)
        
        # Augmentation
        self.pre_augment = Compose([
            SeqPerturb_Reverse(
                method=sp_method,
                num_segments=num_seq_segments,
                fixed_second=fixed_second,
                p=p_ts
            ), # Sequence Perturbation + Reverse

            TimeMaskBack(
                min_band_part=tm_min_band_part,
                max_band_part=tm_max_band_part,
                fade=tm_fade,
                p=p_tm,
                min_mask_start_time=fixed_second,
            ), # Make a randomly chosen part of the audio silent.
        ])
        self.post_augment = Compose([
            PitchShift(
                min_semitones=semitone_range[0], 
                max_semitones=semitone_range[1], 
                p=p_ps
            ), # Pitch Shift # S3T settings: Pitch shift the sound up or down without changing the tempo.
            TimeStretch(
                min_rate=tstr_min_rate,
                max_rate=tstr_max_rate,
                p=p_tstr,
            ), # Time Stretch: Stretch the audio in time without changing the pitch.
        ])

    def __len__(self): #""" Total we got 175698 files * 4 tracks """
        return len(self.bp_listdir)
    
    
    def augment_func(self, x, sample_rate):
        x = self.pre_augment(x, sample_rate=sample_rate)
        x = self.post_augment(x, sample_rate=sample_rate)
        return x
    
    
    def load_segment(self, song_name, seg_idx):
        if self.use_lmdb:
            key = f"{song_name}/{self.stem}_seg_{seg_idx}".encode()
            with self.lmdb_env.begin(write=False) as txn:
                byte_data = txn.get(key)
                if byte_data is None:
                    raise KeyError(f"Key {key} not found in LMDB.")
                x = pickle.loads(byte_data)
        else:
            path = os.path.join(self.data_dir, song_name, f"{self.stem}_seg_{seg_idx}.npy")
            x = np.load(path, mmap_mode='r').copy()
        return x
    

    def __getitem__(self, idx):
        # Load audio data from .npy
        song_name = self.bp_listdir[idx]
        
        if not self.eval_mode:
            segment_count = self.seg_counter[song_name]
            seg_idx = random.randint(0, segment_count - 1)
            x = self.load_segment(song_name, seg_idx)
            x_i = self.augment_func(x, sample_rate=self.sample_rate)
            x_j = self.augment_func(x, sample_rate=self.sample_rate)
            return torch.from_numpy(x_i), torch.from_numpy(x_j), \
                self.label_dict[song_name], song_name
        
        else:
            # Load audio data from .npy from index 0
            x = self.load_segment(song_name, self.eval_id)
            return torch.from_numpy(x), self.label_dict[song_name], song_name



if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="MoCoV2_BP")
    
    config = yaml_config_hook("config/moco_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    
    args = parser.parse_args()
    os.makedirs(args.lmdb_dir, exist_ok=True)
    
    if args.use_lmdb:
        create_lmdb(args.seg_dir, args.lmdb_dir)