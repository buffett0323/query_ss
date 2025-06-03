import os
import torchaudio
import numpy as np
import yaml
import soundfile as sf
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Path
split = "test"  # "test"
path = "/home/buffett/NAS_NTU"
input_path = f"{path}/slakh2100_flac_redux/{split}"
output_path = f"{path}/slakh2100_6stems_npy/{split}"
os.makedirs(output_path, exist_ok=True)

# Number of processes (use all available cores)
NUM_PROCESSES = min(cpu_count(), 24)  # Limit to 24 cores if too many processes slow down disk IO
TARGET_SR = 44100
VALID_STEMS = {"Piano", "Bass", "Guitar", "Drums", "Strings"}


def convert_flac_to_npy(input_flac):
    """Load FLAC file and return waveform."""
    waveform, sr = torchaudio.load(input_flac, format="flac")
    if sr != TARGET_SR:
        print(f"Warning: {input_flac} has sample rate {sr}, expected {TARGET_SR}")
    return waveform


def process_track(track):
    """Process a single track, ensuring only the 6 required stems."""
    yaml_file = os.path.join(input_path, track, "metadata.yaml")
    if not os.path.exists(yaml_file):
        return  # Skip if metadata is missing

    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    stem_mapping = {stem: info["inst_class"] for stem, info in data["stems"].items()}
    stems_path = os.path.join(input_path, track, "stems")
    output_folder = os.path.join(output_path, track)
    os.makedirs(output_folder, exist_ok=True)

    mixed_stems = {stem: [] for stem in VALID_STEMS.union({"Others"})}

    for stem_file in os.listdir(stems_path):
        stem_name, ext = os.path.splitext(stem_file)
        if ext.lower() != ".flac":
            continue

        input_flac = os.path.join(stems_path, stem_file)
        stem_category = stem_mapping.get(stem_name, "Others")


        if "strings" in stem_category.lower():
            mixed_stems["Strings"].append(convert_flac_to_npy(input_flac))
        elif stem_category in VALID_STEMS:
            mixed_stems[stem_category] = [convert_flac_to_npy(input_flac)]
        else:
            mixed_stems["Others"].append(convert_flac_to_npy(input_flac))

    # Save processed stems
    for stem, waveforms in mixed_stems.items():
        if waveforms:
            combined_waveform = sum(waveforms) / len(waveforms)  # Mix multiple sources if needed
            np.save(os.path.join(output_folder, f"{stem}.npy"), combined_waveform.numpy())
        # else:
        #     np.save(os.path.join(output_folder, f"{stem}.npy"), np.zeros((1, 1)))  # Empty placeholder


if __name__ == "__main__":
    tracks = os.listdir(input_path)  # List all tracks in the input path

    with Pool(NUM_PROCESSES) as pool:
        list(tqdm(pool.imap(process_track, tracks), total=len(tracks), desc="Processing Tracks"))

    print("Processing complete.")
