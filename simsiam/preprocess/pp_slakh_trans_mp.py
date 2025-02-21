import os
import torchaudio
import torchaudio.transforms as T
import yaml
import soundfile as sf
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Path
split = "train"
input_path = f"/mnt/gestalt/home/ddmanddman/slakh2100_flac_redux/{split}"
output_path = f"/mnt/gestalt/home/ddmanddman/slakh2100_demucs/{split}"
os.makedirs(output_path, exist_ok=True)

# Number of processes (use all available cores)
NUM_PROCESSES = min(cpu_count(), 8)  # Limit to 8 cores if too many processes slow down disk IO



def convert_flac_to_wav(input_flac, output_wav, target_sr=44100):
    """ Convert FLAC file to WAV with target sample rate. """
    waveform, sr = torchaudio.load(input_flac, format="flac")
    if sr != 44100:
        print("Wrong sr when converting:", sr)    
    
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    
    sf.write(output_wav, waveform.squeeze(0).numpy(), target_sr)



def process_track(track):
    """ Function to process a single track. """
    yaml_file = os.path.join(input_path, track, "metadata.yaml")
    
    if not os.path.exists(yaml_file):
        return  # Skip if metadata is missing

    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
    
    stem_mapping = {stem: info["inst_class"] for stem, info in data["stems"].items()}
    
    for stem in os.listdir(os.path.join(input_path, track, "stems")):
        input_flac = os.path.join(input_path, track, "stems", stem)
        output_wav_folder = os.path.join(output_path, track)
        os.makedirs(output_wav_folder, exist_ok=True)

        # Get base name from YAML mapping
        basename = stem_mapping[os.path.splitext(stem)[0]]
        output_wav = os.path.join(output_wav_folder, f"{basename}.wav")

        # Convert FLAC to WAV
        convert_flac_to_wav(input_flac, output_wav)



if __name__ == "__main__":
    tracks = os.listdir(input_path)  # List all tracks in the input path

    # Use multiprocessing Pool to process tracks in parallel
    with Pool(NUM_PROCESSES) as pool:
        list(tqdm(pool.imap(process_track, tracks), total=len(tracks), desc="Processing Tracks"))

    print("Processing complete.")
