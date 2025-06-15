import os
import glob
import numpy as np
import torch
import torchaudio
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

# Constants
SAMPLE_RATE = 44100
TARGET_DURATION = 2.97  # seconds
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION)  # 130,977 samples

def process_file(file, output_path):
    try:
        wav, sr = torchaudio.load(file)
        
        # Ensure correct sample rate
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        
        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Pad or truncate to target length
        current_length = wav.shape[1]
        if current_length < TARGET_SAMPLES:
            # Pad with zeros
            padding = TARGET_SAMPLES - current_length
            wav = torch.nn.functional.pad(wav, (0, padding))
        elif current_length > TARGET_SAMPLES:
            # Truncate
            wav = wav[:, :TARGET_SAMPLES]
        
        # Save as numpy array
        output_file = os.path.join(output_path, os.path.basename(file).replace(".wav", ".npy"))
        np.save(output_file, wav.numpy())
        return True
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        return False

def main():
    input_path = "/mnt/gestalt/home/buffett/rendered_adsr_dataset"
    output_path = "/mnt/gestalt/home/buffett/rendered_adsr_dataset_npy"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Get list of all wav files
    wav_files = glob.glob(os.path.join(input_path, "*.wav"))
    print(f"Found {len(wav_files)} WAV files to process")
    print(f"Target duration: {TARGET_DURATION}s ({TARGET_SAMPLES} samples at {SAMPLE_RATE}Hz)")
    
    # Create process pool
    num_workers = mp.cpu_count()
    print(f"Using {num_workers} workers for parallel processing")
    
    # Create partial function with fixed output path
    process_func = partial(process_file, output_path=output_path)
    
    # Process files in parallel
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, wav_files),
            total=len(wav_files),
            desc="Converting WAV to NPY"
        ))
    
    # Report results
    successful = sum(results)
    print(f"Successfully converted {successful}/{len(wav_files)} files")

if __name__ == '__main__':
    main()