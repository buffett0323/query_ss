import numpy as np
import pandas as pd
import seaborn as sns
import os
import glob
import torchaudio as ta
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from tqdm import tqdm
from resemblyzer import VoiceEncoder, preprocess_wav, normalize_volume, trim_long_silences
from pathlib import Path
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from scipy.io import wavfile


NAS_NAME = "NAS_NTU"
REV_NAME = "" #"_6sec"
QUERY_NAME = ".query-10s"
QUERY_ROOT = f"/home/buffett/{NAS_NAME}/moisesdb/npyq{REV_NAME}"
BEATS_ROOT = f"/home/buffett/{NAS_NAME}/moisesdb/beats{REV_NAME}"


# Get the BEATs informed data
beats = glob.glob(os.path.join(BEATS_ROOT, "**", f"*.beats{REV_NAME}.npy"), recursive=True)

BEATS_path = []
ORIG_mixture = []
ORIG_target = []
stems = ["vocals", "bass", "drums", "vdbo_others"] # VBDO Settings



print("----- Loading d-vector embedding from BEATs pre-trained models -----")
for beats_file in tqdm(beats):
    if f"mixture.beats{REV_NAME}.npy" in beats_file: continue

    target_path = beats_file.replace(f"/beats{REV_NAME}",f"/npyq{REV_NAME}").replace(f".beats{REV_NAME}.npy", f"{QUERY_NAME}.npy") #.npy
    mixture_path = target_path.replace(target_path.split('/')[-1], f"mixture{QUERY_NAME}.npy") # mixture.npy

    stem = str(beats_file).split('/')[-1].split('.')[0]
    if stem not in stems:
        continue# stems.append(stem)
    
    BEATS_path.append(beats_file)
    ORIG_target.append(target_path)
    ORIG_mixture.append(mixture_path)
    
   
print("Get BEATs Dataset stem count: ", len(BEATS_path), "out of", len(beats))

# Get the unique stems
stems = set(stems)
# print(f"Use {len(stems)} Stems: {stems}")
