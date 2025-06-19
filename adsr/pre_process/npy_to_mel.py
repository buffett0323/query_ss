import os
import numpy as np
import torch
from tqdm import tqdm
import nnAudio.features
import glob

def process_batch(files, input_path, output_path, batch_size=64):
    """Process files in batches for better efficiency"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize mel spectrogram converter once
    to_spec = nnAudio.features.MelSpectrogram(
        sr=44100,
        n_mels=128,
        fmin=20,
        fmax=22050,
        hop_length=512,
        n_fft=2048,
        window='hann',
        center=True,
        power=2.0,
    ).to(device)

    # Calculate total batches for progress bar
    (len(files) + batch_size - 1) // batch_size

    successful = 0
    with tqdm(total=len(files), desc="Processing files") as pbar:
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]

            # Load all audio files in batch
            wavs = []
            valid_files = []

            for file in batch_files:
                try:
                    wav = np.load(os.path.join(input_path, file))
                    wavs.append(wav)
                    valid_files.append(file)
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")
                    pbar.update(1)
                    continue

            if not wavs:
                pbar.update(len(batch_files))
                continue

            # Stack into batch tensor
            try:
                wav_batch = torch.from_numpy(np.stack(wavs)).to(device)

                # Convert to mel spectrograms in batch
                with torch.no_grad():  # Disable gradients for inference
                    mel_batch = to_spec(wav_batch)

                # Save each mel spectrogram
                for j, file in enumerate(valid_files):
                    mel = mel_batch[j].cpu().numpy()
                    output_file = os.path.join(output_path, file.replace(".npy", "_mel.npy"))
                    np.save(output_file, mel)
                    successful += 1

                pbar.update(len(batch_files))

            except Exception as e:
                print(f"Error processing batch starting with {batch_files[0]}: {str(e)}")
                pbar.update(len(batch_files))
                continue

    return successful

if __name__ == "__main__":
    input_path = "/mnt/gestalt/home/buffett/rendered_adsr_dataset_npy"
    output_path = "/mnt/gestalt/home/buffett/rendered_adsr_dataset_npy_new_mel"

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Get all files using glob (faster than os.listdir)
    files = glob.glob(os.path.join(input_path, "*.npy"))
    files = [os.path.basename(f) for f in files]  # Get just filenames

    print(f"Found {len(files)} files to process")

    # Process in batches
    batch_size = 64  # Larger batch size for better GPU utilization
    successful = process_batch(files, input_path, output_path, batch_size)

    print(f"Successfully processed {successful} files")
    print(f"Failed to process {len(files) - successful} files")
