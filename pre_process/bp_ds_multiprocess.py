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
output_path = f"{path}/beatport_analyze/chorus_audio_wav"
json_folder = f"{path}/beatport_analyze/json"
htdemucs_folder = f"{path}/beatport_analyze/htdemucs"
os.makedirs(output_path, exist_ok=True)
THRES = 8
SR = 44100

def process_file(js):
    with open(os.path.join(json_folder, js), "r", encoding="utf-8") as file:
        data = json.load(file)

    data_path = data["path"]
    y, sr = torchaudio.load(data_path)

    # Process segments labeled "chorus"
    counter = 0
    name = os.path.basename(data_path).replace(".mp3", "")
    for segment in data["segments"]:
        if segment["label"] == "chorus" and (segment["end"] - segment["start"]) > THRES:
            counter += 1
            start_sample = int(segment["start"] * sr)  # Convert seconds to samples
            end_sample = int(segment["end"] * sr)

            # Slice the waveform
            mix_seg = y[:, start_sample:end_sample]

            # Save as .npy file
            segment_folder = os.path.join(output_path, f"{name}_{counter}")
            os.makedirs(segment_folder, exist_ok=True)

            # Save the mix
            torchaudio.save(f"{segment_folder}/mix.wav", mix_seg, SR, format="wav")

            # Mix bass and other stems together
            bass_seg = torchaudio.load(os.path.join(htdemucs_folder, name, "bass.wav"))[0][:, start_sample:end_sample]
            other_seg = torchaudio.load(os.path.join(htdemucs_folder, name, "other.wav"))[0][:, start_sample:end_sample]
            drums_seg = torchaudio.load(os.path.join(htdemucs_folder, name, "drums.wav"))[0][:, start_sample:end_sample]


            torchaudio.save(f"{segment_folder}/bass.wav", bass_seg, SR, format="wav")
            torchaudio.save(f"{segment_folder}/other.wav", other_seg, SR, format="wav")
            torchaudio.save(f"{segment_folder}/drums.wav", drums_seg, SR, format="wav")

            bass_other_mix = bass_seg + other_seg
            torchaudio.save(f"{segment_folder}/bass_other.wav", bass_other_mix, SR, format="wav")


if __name__ == '__main__':
    # Get list of all JSON files
    json_files = os.listdir(json_folder)[:10]

    # Create a pool of workers
    num_processes = cpu_count() - 2  # Leave one CPU free
    pool = Pool(processes=num_processes)

    # Process files in parallel with progress bar
    list(tqdm(pool.imap(process_file, json_files), total=len(json_files)))

    # Close the pool
    pool.close()
    pool.join()
