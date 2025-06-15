import os
import librosa
import torchaudio
import torchaudio.transforms as T
import numpy as np
from tqdm import tqdm
import soundfile as sf
import yaml


# path
input_path = "/mnt/gestalt/home/ddmanddman/slakh2100_flac_redux/train"
output_path = "/mnt/gestalt/home/ddmanddman/slakh2100_buffett/train"
os.makedirs(output_path, exist_ok=True)


def convert_flac_to_npy(input_flac, target_sr=16000):
    # Load FLAC file
    waveform, sr = torchaudio.load(input_flac, format="flac")
    if sr != 44100:
        print("Wrong sr when converting:", sr)

    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform.numpy()


def energy_activity_detection(audio, output_path, basename, sr=16000):
    # Use librosa to detect energy-based activity
    non_silent_intervals = librosa.effects.split(audio, top_db=30)  # Detect non-silent regions

    # Find a valid 4-second segment
    segment_duration = sr * 4  # 4 seconds in samples
    selected_segment = None

    for start, end in non_silent_intervals:
        if (end - start) >= segment_duration:
            selected_segment = audio[start : start + segment_duration]
            break


    if selected_segment is not None:

        # Save to .npy
        output_npy = os.path.join(output_path, f"{basename}.npy")
        np.save(output_npy, selected_segment)

        # Save to .wav
        output_wav = os.path.join(output_path, f"{basename}.wav")
        sf.write(output_wav, selected_segment, sr)



if __name__ == "__main__":


    label_set = set()
    for track in tqdm(os.listdir(input_path), desc="Converting Flac"):

        # Load the YAML file
        yaml_file = os.path.join(input_path, track, "metadata.yaml")
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
        stem_mapping = {stem: info["inst_class"] for stem, info in data["stems"].items()}


        for stem in os.listdir(os.path.join(input_path, track, "stems")):
            input_flac = os.path.join(input_path, track, "stems", stem)
            output_wav_folder = os.path.join(output_path, track)
            os.makedirs(output_wav_folder, exist_ok=True)

            # Get base name
            basename = stem_mapping[os.path.splitext(os.path.basename(input_flac))[0]]
            label_set.add(basename)


            # Step 2 Convert flac to wav
            wav_npy = convert_flac_to_npy(input_flac) # (1, 4082068)

            if len(wav_npy.shape) > 1:
                wav_npy = wav_npy.squeeze(0)

            # Step 3 Energy-based activity detection
            energy_activity_detection(wav_npy, output_wav_folder, basename=basename)




    with open('../info/slakh_label.txt', 'w') as f:
        for l in list(label_set):
            f.write(f"{l}\n")
