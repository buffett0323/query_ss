# Not used
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader

SET_LENGTH = 65280

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, mixture_files, embedding_files, target_files):
        self.mixture_files = mixture_files
        self.embedding_files = embedding_files
        self.target_files = target_files

    def __len__(self):
        return len(self.mixture_files)

    def __getitem__(self, idx):
        # Load the .npy files using numpy
        mixture = np.load(self.mixture_files[idx])#[:SET_LENGTH]
        embedding = np.load(self.embedding_files[idx]).squeeze(0) # torch.Size([4, 1376, 768])
        target = np.load(self.target_files[idx])#[:SET_LENGTH]
        
        
        # # Added for solving 2D into 1D
        # mixture = mixture.mean(axis=0)[:SET_LENGTH]
        # target = target.mean(axis=0)[:SET_LENGTH]
        
        # Convert the loaded numpy arrays to torch tensors
        mixture = torch.tensor(mixture, dtype=torch.float32)
        embedding = torch.tensor(embedding, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        stem = str(self.embedding_files[idx]).split('/')[-1].split('.')[0]

        return mixture, embedding, target, stem
