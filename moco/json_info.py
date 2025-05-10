import os
import json
from tqdm import tqdm

dic = {}
path = "/mnt/gestalt/home/buffett/beatport_analyze/chorus_audio_16000_095sec_npy_bass_other_new/amp_08"
for file in tqdm(os.listdir(path)):
    dic[file] = len(os.listdir(os.path.join(path, file)))

with open("info/train_seg_counter_amp08.json", "w") as f:
    json.dump(dic, f)