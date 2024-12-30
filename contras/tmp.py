# import os
# import random
# import shutil
# import torch
# import torchaudio
# import argparse
# import torch.nn as nn
# import torch.multiprocessing as mp
# import torchaudio.transforms as T
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from mutagen.mp3 import MP3
# from utils import yaml_config_hook


# class BeatportDataset(Dataset):
#     def __init__(
#         self, 
#         dataset_dir, 
#         args,
#         device='cpu',
#         split="train", 
#         n_fft=2048, 
#         hop_length=1024,
#     ):
#         """
#         Args:
#             dataset_dir (str): Path to the dataset directory.
#             split (str): The split to load ('train', 'test', 'unlabeled').
#         """
#         self.dataset_dir = dataset_dir
#         self.my_device = device
#         self.split = split
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.target_length = n_fft + hop_length
#         self.max_length = args.max_length
        
#         self.audio_files = [
#             os.path.join(dataset_dir, folder_name, file_name)
#             for folder_name in os.listdir(dataset_dir)
#                 for file_name in os.listdir(os.path.join(dataset_dir, folder_name))
#                     if file_name.endswith(".mp3")
#         ]
#         self.preprocessed_dir = '/mnt/gestalt/home/ddmanddman/beatport_preprocess/pt'
#         os.makedirs(self.preprocessed_dir, exist_ok=True)

#     def preprocess_file(self, file_name):
#         """Preprocess a single audio file."""
#         try:
#             if file_name.endswith(".mp3"):
#                 song_name = file_name.split('/')[-2]
#                 waveform, _ = torchaudio.load(file_name)
#                 os.makedirs(os.path.join(self.preprocessed_dir, song_name), exist_ok=True)
#                 save_path = os.path.join(
#                     self.preprocessed_dir, 
#                     song_name, 
#                     f"{file_name.split('/')[-1].split('.mp3')[0]}.pt"
#                 )
#                 torch.save(waveform, save_path)
#         except Exception as e:
#             print(f"Error processing file {file_name}: {e}")

#     def preprocess(self, num_workers=24):
#         """Preprocess audio files using multiprocessing."""
#         with mp.Pool(processes=num_workers) as pool:
#             list(tqdm(pool.imap(self.preprocess_file, self.audio_files), total=len(self.audio_files)))


# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)  # Ensure multiprocessing uses 'spawn'

#     parser = argparse.ArgumentParser(description="SimCLR")

#     config = yaml_config_hook("config.yaml")
#     for k, v in config.items():
#         parser.add_argument(f"--{k}", default=v, type=type(v))

#     args = parser.parse_args()
#     dataset_dir = args.dataset_dir
#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")

#     # Initialize Dataset and preprocess
#     train_dataset = BeatportDataset(
#         dataset_dir=args.dataset_dir,
#         args=args,
#         split="train",
#         n_fft=args.n_fft, 
#         hop_length=args.hop_length,
#     )

#     print("Starting preprocessing...")
#     train_dataset.preprocess(num_workers=24)  # Use 8 workers for parallel processing
#     print("Preprocessing complete.")

import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

path = "/mnt/gestalt/home/ddmanddman/beatport_analyze/spec"


class DBDS(Dataset):
    def __init__(self):
        super(DBDS).__init__()
        self.lis = os.listdir(path)
    
    def __getitem__(self, idx):
        return np.load(os.path.join(path, self.lis[idx]))
    
    def __len__(self):
        return len(self.lis)
    
# ds = DBDS()
# print(len(ds))

# train_loader = DataLoader(ds, batch_size=128, num_workers=24, pin_memory=True, prefetch_factor=2, )
# for i in tqdm(train_loader):
#     pass

pp = '/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_npy/05972218-2454-43cc-a7c9-4fd725590e7c_1/vocals_1_mel.npy'
ppp = np.load(pp)
print(ppp.shape)