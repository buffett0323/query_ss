import os
import json
import librosa
import torchaudio
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Init settings
# path = "/home/buffett/NAS_NTU" #
path = "/mnt/gestalt/home/ddmanddman"
output_path = f"{path}/beatport_analyze/chorus_audio_npy"
json_folder = f"{path}/beatport_analyze/json"
htdemucs_folder = f"{path}/beatport_analyze/htdemucs"
os.makedirs(output_path, exist_ok=True)
thres = 6
sr = 44100

for js in tqdm(os.listdir(json_folder)):
    
    with open(os.path.join(json_folder, js), "r", encoding="utf-8") as file:
        data = json.load(file)  # Load JSON content

    # Print the JSON content
    # print(json.dumps(data))#, indent=4)) 

    data_path = data["path"]
    # data_path = data_path.replace("/mnt/gestalt/database", "/home/buffett/NAS_DB")
    y, sr = torchaudio.load(data_path)

    # Process segments labeled "chorus"
    counter = 0
    name = os.path.basename(data_path).replace(".mp3", "")
    for segment in data["segments"]:
        if segment["label"] == "chorus" and (segment["end"] - segment["start"]) > thres:
            counter += 1
            start_sample = int(segment["start"] * sr)  # Convert seconds to samples
            end_sample = int(segment["end"] * sr)

            # Slice the waveform
            mix_seg = y[:, start_sample:end_sample]

            # Save as .npy file
            segment_folder = os.path.join(output_path, f"{name}_{counter}")
            os.makedirs(segment_folder, exist_ok=True)
            
            # Save the mix
            np.save(f"{segment_folder}/mix.npy", mix_seg)
            
            for stem in ["drums", "bass", "other", "vocals"]:
                ht_stem, _ = torchaudio.load(os.path.join(htdemucs_folder, name, f"{stem}.wav"))#, sr=sr)
                stem_seg = ht_stem[:, start_sample:end_sample]
                np.save(f"{segment_folder}/{stem}.npy", stem_seg)


    break