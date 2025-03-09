import os
import json
import librosa
import torchaudio
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Init settings
section = "verse"
path = "/mnt/gestalt/home/ddmanddman" # "/home/buffett/NAS_NTU" # 
output_path = f"{path}/beatport_analyze/{section}_audio_npy" # f"{path}/beatport_analyze/chorus_audio_npy"
json_folder = f"{path}/beatport_analyze/json"
htdemucs_folder = f"{path}/beatport_analyze/htdemucs"
stems = ["vocals"] #["drums", "bass", "other", "vocals"]

os.makedirs(output_path, exist_ok=True)
thres = 8
sr = 44100
num_workers = max(1, cpu_count() - 2)  # Use available CPU cores minus 2 to avoid system freeze
print("Num workers:", num_workers)


# Get list of JSON files
json_files = [os.path.join(json_folder, js) for js in os.listdir(json_folder)]


def process_json(json_path):
    """
    Function to process a single JSON file: load the JSON, extract chorus segments,
    save mix and stems as .npy.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Load the original audio
        data_path = data["path"]
        # data_path = data_path.replace("/mnt/gestalt/database", "/home/buffett/NAS_DB")
        y, _ = torchaudio.load(data_path)#, sr=sr)

        # Extract name from path
        name = os.path.basename(data_path).replace(".mp3", "")

        counter = 0
        for segment in data["segments"]:
            if segment["label"] == section and (segment["end"] - segment["start"]) > thres:
                counter += 1
                start_sample = int(segment["start"] * sr)  # Convert time to samples
                end_sample = int(segment["end"] * sr)

                # Slice the waveform
                mix_seg = y[:, start_sample:end_sample]

                # Create folder for the segment
                segment_folder = os.path.join(output_path, f"{name}_{counter}")
                os.makedirs(segment_folder, exist_ok=True)

                # Save the mix segment
                np.save(f"{segment_folder}/mix.npy", mix_seg)

                # Load stems from htdemucs folder and save
                for stem in stems:
                    stem_path = os.path.join(htdemucs_folder, name, f"{stem}.wav")
                    if os.path.exists(stem_path):  # Ensure the stem file exists
                        ht_stem, _ = torchaudio.load(stem_path)
                        stem_seg = ht_stem[:, start_sample:end_sample]
                        np.save(f"{segment_folder}/{stem}.npy", stem_seg)

        return f"Processed: {json_path}"

    except Exception as e:
        return f"Error processing {json_path}: {str(e)}"


# Run in parallel using multiprocessing
if __name__ == "__main__":
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_json, json_files), total=len(json_files)))

    # Print summary
    for result in results:
        print(result)