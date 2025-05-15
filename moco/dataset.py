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
import multiprocessing

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
from utils import yaml_config_hook, create_lmdb
# import audiomentations
# print("Audiomentations Path:", audiomentations.__file__)




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
        amp_name="_amp_05",
        loading_mode="simple",#"pairs",
    ):
        # Load segment info list
        print(f"Loading {split} segment counter from {amp_name}, with {loading_mode} mode")
        
        with open(f"info/{split}_segments{amp_name}.json", "r") as f:
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
        self.loading_mode = loading_mode
        
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
    
    
    def load_segment(self, song_name):
        # TODO: Random Choose Segment
        # if self.eval_mode:
        #     seg_idx = self.eval_id
        # else:
        #     segment_count = self.seg_counter[song_name]
        #     seg_idx = random.randint(0, segment_count - 1)
        seg_idx = self.eval_id
        path = os.path.join(self.data_dir, song_name, f"{self.stem}_seg_{seg_idx}.npy")
        x = np.load(path, mmap_mode='r').copy()
        return x
    
    
    def load_pairs(self, song_name):
        segment_count = self.seg_counter[song_name]
        seg_idx1, seg_idx2 = random.sample(range(segment_count), 2)
        path1 = os.path.join(self.data_dir, song_name, f"{self.stem}_seg_{seg_idx1}.npy")
        path2 = os.path.join(self.data_dir, song_name, f"{self.stem}_seg_{seg_idx2}.npy")
        x_pair1 = np.load(path1, mmap_mode='r').copy()
        x_pair2 = np.load(path2, mmap_mode='r').copy()
        return x_pair1, x_pair2


    def __getitem__(self, idx):
        # Load audio data from .npy
        song_name = self.bp_listdir[idx]
        
        if not self.eval_mode:
            if self.loading_mode == "simple":
                x = self.load_segment(song_name)
                x_i = self.augment_func(x, sample_rate=self.sample_rate)
                x_j = self.augment_func(x, sample_rate=self.sample_rate)
                return torch.from_numpy(x_i), torch.from_numpy(x_j), \
                        self.label_dict[song_name], song_name
            
            elif self.loading_mode == "pairs":
                x_pair1, x_pair2 = self.load_pairs(song_name)
                x_i = self.augment_func(x_pair1, sample_rate=self.sample_rate)
                x_j = self.augment_func(x_pair1, sample_rate=self.sample_rate)
                return torch.from_numpy(x_pair1), torch.from_numpy(x_pair2), \
                        torch.from_numpy(x_i), torch.from_numpy(x_j)
                        
            else:
                raise ValueError(f"Invalid loading mode: {self.loading_mode}")
        
        else:
            # Load audio data from .npy from index 0
            x = self.load_segment(song_name)
            return torch.from_numpy(x), self.label_dict[song_name], song_name




if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="MoCoV2_BP")
    
    config = yaml_config_hook("config/moco_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    
    args = parser.parse_args()
    os.makedirs(args.lmdb_dir, exist_ok=True)
    
    # if args.use_lmdb:
    #     create_lmdb(args.seg_dir, args.lmdb_dir)
    
    train_dataset = SegmentBPDataset(
        data_dir=args.seg_dir,
        split="train",
        stem="bass_other",
        eval_mode=False,
        num_seq_segments=args.num_seq_segments,
        fixed_second=args.fixed_second,
        sp_method=args.sp_method,
        p_ts=args.p_ts,
        p_ps=args.p_ps,
        p_tm=args.p_tm,
        p_tstr=args.p_tstr,
        semitone_range=args.semitone_range,
        tm_min_band_part=args.tm_min_band_part,
        tm_max_band_part=args.tm_max_band_part,
        tm_fade=args.tm_fade,
        amp_name=args.amp_name,
    )
    
    print("train_dataset length:", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8, #args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    
    import time
    time_start = time.time()
    for i, (x_i, x_j, label, song_name) in enumerate(tqdm(train_loader)):
        # print(x_i.shape, x_j.shape, label, song_name)
        # break
        if i > 100:
            break
    time_end = time.time()
    print("time cost:", time_end - time_start)
    
    # print(train_loader.dataset[0])