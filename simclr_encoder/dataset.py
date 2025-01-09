# Using original track, without .dbfss
import os
import random
import shutil
import torch
import torchaudio
import argparse
import numpy as np
import torch.nn as nn
import torch.multiprocessing as mp
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram
from pytorch_lightning import LightningDataModule
from typing import Optional
from tqdm import tqdm
from mutagen.mp3 import MP3




class MoisesDBDataset(Dataset):
    def __init__(
        self, 
        data_path_list,
        split="train",
    ):
        self.split = split
        self.data_path_list = data_path_list
        

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        path = self.data_path_list[idx]
        file = np.load(path)
        



if __name__ == "__main__":
    path = "/mnt/gestalt/home/ddmanddman/moisesdb/npy2/"
    dp_list = [
        os.path.join(path, folder_name, file_name)
        for folder_name in os.listdir(path)
            for file_name in os.listdir(os.path.join(path, folder_name))
                if file_name.endswith(".npy") and not file_name.endswith(".dbfs.npy") 
    ]
    
    random.shuffle(dp_list)
    n = len(dp_list)
    train_end = int(n * 0.8)  # 80%
    test_end = train_end + int(n * 0.1)  # 80% + 10%

    # Split the data
    train_data = dp_list[:train_end]
    test_data = dp_list[train_end:test_end]
    valid_data = dp_list[test_end:]
    db = MoisesDBDataset(data_path_list=train_data)
    print(len(train_data))