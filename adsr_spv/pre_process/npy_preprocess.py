import json
import torch
import torchaudio
import numpy as np
import nnAudio.features
from pathlib import Path
from tqdm import tqdm

def process_audio_batch(
    batch_data: list,
    mel_converter: nnAudio.features.MelSpectrogram,
    unit_length: int,
    n_mels: int,
    expected_mel_frames: int,
    device: str,
    output_dir: Path,
    split: str
):
    """
    Process a batch of audio files and save mel spectrograms as .npy files.

    Args:
        batch_data: List of (idx, chunk) tuples
        mel_converter: Pre-initialized mel spectrogram converter
        unit_length: Target audio length in samples
        n_mels: Number of mel frequency bins
        expected_mel_frames: Expected number of time frames
        device: Device to run computation on
        output_dir: Directory to save .npy files
        split: Data split name
    """
    results = []

    # Extract batch information
    [item[0] for item in batch_data]
    [item[1] for item in batch_data]
    batch_paths = [item[1]["file_path"] for item in batch_data]

    try:
        # Load all audio files in the batch
        audio_batch = []
        for wav_path in batch_paths:
            wav, _ = torchaudio.load(wav_path)
            wav = wav.mean(dim=0, keepdim=True)  # Convert to mono
            audio_batch.append(wav)

        # Stack all audio tensors into a single batch
        # Pad or truncate each audio to unit length
        processed_audio = []
        for wav in audio_batch:
            current_length = wav.shape[1]
            if current_length < unit_length:
                # Pad with zeros
                padding = unit_length - current_length
                wav = torch.nn.functional.pad(wav, (0, padding))
            elif current_length > unit_length:
                # Truncate
                wav = wav[:, :unit_length]
            processed_audio.append(wav)

        # Stack into batch tensor [batch_size, 1, unit_length]
        audio_batch_tensor = torch.stack(processed_audio, dim=0)

        # Move to device and convert to mel spectrograms in batch
        audio_batch_tensor = audio_batch_tensor.to(device)
        mel_batch = mel_converter(audio_batch_tensor)  # [batch_size, n_mels, time]

        # Process each mel spectrogram in the batch
        for i, (idx, chunk) in enumerate(batch_data):
            mel = mel_batch[i]  # [n_mels, time]

            # Ensure correct shape
            if mel.shape[-1] > expected_mel_frames:
                mel = mel[:, :expected_mel_frames]
            elif mel.shape[-1] < expected_mel_frames:
                # Pad with zeros
                padding = expected_mel_frames - mel.shape[-1]
                mel = torch.nn.functional.pad(mel, (0, padding))

            # Convert to numpy and save
            mel_np = mel.cpu().numpy().astype(np.float32)

            # Save as .npy file
            npy_filename = f"{idx:06d}_{chunk['file'].replace('.wav', '_mel.npy')}"
            npy_path = output_dir / split / npy_filename
            np.save(npy_path, mel_np)

            # Store metadata
            result = {
                'idx': idx,
                'file': chunk['file'],
                'npy_file': npy_filename,
                'attack': chunk['attack'],
                'decay': chunk['decay'],
                'sustain': chunk['sustain'],
                'release': chunk['release'],
                'mel_shape': mel_np.shape
            }
            results.append(result)

    except Exception as e:
        print(f"Error processing batch: {e}")
        # Fallback to individual processing
        for idx, chunk in batch_data:
            try:
                # Load audio
                wav_path = chunk["file_path"]
                wav, _ = torchaudio.load(wav_path)
                wav = wav.mean(dim=0, keepdim=True)  # Convert to mono

                # Pad or truncate to unit length
                current_length = wav.shape[1]
                if current_length < unit_length:
                    # Pad with zeros
                    padding = unit_length - current_length
                    wav = torch.nn.functional.pad(wav, (0, padding))
                elif current_length > unit_length:
                    # Truncate
                    wav = wav[:, :unit_length]

                # Convert to mel spectrogram on GPU
                mel = mel_converter(wav.to(device))

                # Ensure correct shape
                if mel.shape[-1] > expected_mel_frames:
                    mel = mel[:, :expected_mel_frames]
                elif mel.shape[-1] < expected_mel_frames:
                    # Pad with zeros
                    padding = expected_mel_frames - mel.shape[-1]
                    mel = torch.nn.functional.pad(mel, (0, padding))

                # Convert to numpy and save
                mel_np = mel.cpu().numpy().astype(np.float32)

                # Save as .npy file
                npy_filename = f"{idx:06d}_{chunk['file'].replace('.wav', '_mel.npy')}"
                npy_path = output_dir / split / npy_filename
                np.save(npy_path, mel_np)

                # Store metadata
                result = {
                    'idx': idx,
                    'file': chunk['file'],
                    'npy_file': npy_filename,
                    'attack': chunk['attack'],
                    'decay': chunk['decay'],
                    'sustain': chunk['sustain'],
                    'release': chunk['release'],
                    'mel_shape': mel_np.shape
                }
                results.append(result)

            except Exception as e2:
                print(f"Error processing file {chunk['file']} at index {idx}: {e2}")
                # Create empty mel spectrogram
                mel_np = np.zeros((n_mels, expected_mel_frames), dtype=np.float32)
                npy_filename = f"{idx:06d}_{chunk['file'].replace('.wav', '_mel.npy')}"
                npy_path = output_dir / split / npy_filename
                np.save(npy_path, mel_np)

                result = {
                    'idx': idx,
                    'file': chunk['file'],
                    'npy_file': npy_filename,
                    'attack': chunk['attack'],
                    'decay': chunk['decay'],
                    'sustain': chunk['sustain'],
                    'release': chunk['release'],
                    'mel_shape': mel_np.shape
                }
                results.append(result)

    return results



def create_mel_npy_dataset(
    data_dir: Path,
    output_dir: Path,
    split: str = "train",
    unit_sec: float = 2.97,
    sr: int = 44100,
    n_mels: int = 128,
    fmin: int = 20,
    fmax: int = 22050,
    hop_length: int = 512,
    n_fft: int = 2048,
    window: str = 'hann',
    center: bool = True,
    power: float = 2.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
    num_workers: int = 4
):
    """
    Convert audio waveforms to mel spectrograms and store as .npy files efficiently.

    Args:
        data_dir: Directory containing the audio data
        output_dir: Directory to save the .npy files
        split: Data split (train/val/test)
        unit_sec: Duration of each audio segment in seconds
        sr: Sample rate
        n_mels: Number of mel frequency bins
        fmin: Minimum frequency
        fmax: Maximum frequency
        hop_length: Hop length for STFT
        n_fft: FFT window size
        window: Window function type
        center: Whether to center the STFT
        power: Power of the spectrogram
        device: Device to run mel spectrogram computation on
        batch_size: Number of files to process in each batch
        num_workers: Number of worker processes
    """

    # Create output directory
    output_split_dir = output_dir / split
    output_split_dir.mkdir(parents=True, exist_ok=True)

    # Initialize mel spectrogram converter
    mel_converter = nnAudio.features.MelSpectrogram(
        sr=sr,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        hop_length=hop_length,
        n_fft=n_fft,
        window=window,
        center=center,
        power=power,
    ).to(device)

    # Load metadata
    metadata_path = data_dir / split / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)


    # Add file paths to metadata
    for chunk in metadata:
        chunk["file_path"] = str(data_dir / split / chunk["file"])

    # Calculate expected mel spectrogram dimensions
    unit_length = int(unit_sec * sr)
    expected_mel_frames = 256  # Fixed size for consistency

    print(f"Processing {split} split...")
    print(f"Expected mel frames: {expected_mel_frames}")
    print(f"Unit length: {unit_length}")
    print(f"Number of samples: {len(metadata)}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")

    # Create batches
    batches = []
    for i in range(0, len(metadata), batch_size):
        batch_data = [(j, metadata[j]) for j in range(i, min(i + batch_size, len(metadata)))]
        batches.append(batch_data)

    print(f"Created {len(batches)} batches")

    # Process batches
    all_results = []

    for batch_idx, batch_data in enumerate(tqdm(batches, desc=f"Processing {split} batches")):
        # Process batch on GPU
        batch_results = process_audio_batch(
            batch_data=batch_data,
            mel_converter=mel_converter,
            unit_length=unit_length,
            n_mels=n_mels,
            expected_mel_frames=expected_mel_frames,
            device=device,
            output_dir=output_dir,
            split=split
        )
        all_results.extend(batch_results)

    # Save metadata
    metadata_output_path = output_split_dir / "metadata.json"
    with open(metadata_output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save processing parameters
    params = {
        'unit_sec': unit_sec,
        'sr': sr,
        'n_mels': n_mels,
        'fmin': fmin,
        'fmax': fmax,
        'hop_length': hop_length,
        'n_fft': n_fft,
        'window': window,
        'center': center,
        'power': power,
        'expected_mel_frames': expected_mel_frames,
        'total_samples': len(all_results)
    }

    params_path = output_split_dir / "params.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"Created .npy dataset at {output_split_dir}")
    print(f"Total samples: {len(all_results)}")
    print(f"Metadata saved to: {metadata_output_path}")
    print(f"Parameters saved to: {params_path}")

    return output_split_dir


def create_all_splits(
    data_dir: Path,
    output_dir: Path,
    splits: list = ["val", "test", "train"],
    **kwargs
):
    """Create .npy datasets for all splits."""

    for split in splits:
        print(f"\nProcessing {split} split...")
        try:
            create_mel_npy_dataset(data_dir, output_dir, split, **kwargs)
            print(f"Successfully processed {split} split")
        except FileNotFoundError:
            print(f"Warning: {split} split not found, skipping...")
        except Exception as e:
            print(f"Error processing {split} split: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    BASE_DIR = "/mnt/gestalt/home/buffett" #"/home/buffett/dataset"
    data_dir = Path(f"{BASE_DIR}/rendered_adsr_unpaired")
    output_dir = Path(f"{BASE_DIR}/rendered_adsr_unpaired_mel_npy")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mel spectrogram parameters
    mel_params = {
        'unit_sec': 2.97,
        'sr': 44100,
        'n_mels': 128,
        'fmin': 20,
        'fmax': 22050,
        'hop_length': 512,
        'n_fft': 2048,
        'window': 'hann',
        'center': True,
        'power': 2.0,
        'device': device,
        'batch_size': 32,  # Process 32 files at a time
        'num_workers': 4
    }

    # Create .npy datasets for all splits
    create_all_splits(data_dir, output_dir, **mel_params)
