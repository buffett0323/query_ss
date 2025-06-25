import json
import h5py
import torch
import torchaudio
import numpy as np
import nnAudio.features
from pathlib import Path
from tqdm import tqdm

def create_mel_h5_dataset(
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Convert audio waveforms to mel spectrograms and store in HDF5 format.

    Args:
        data_dir: Directory containing the audio data
        output_dir: Directory to save the HDF5 files
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
    """

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Calculate expected mel spectrogram dimensions
    unit_length = int(unit_sec * sr)
    expected_mel_frames = 256 #int(np.ceil(unit_length / hop_length)) + 1 if center else int(np.ceil(unit_length / hop_length))

    print(f"Expected mel frames: {expected_mel_frames}")
    print(f"Unit length: {unit_length}")
    print(f"Number of samples: {len(metadata)}")

    # Create HDF5 file
    h5_path = output_dir / f"{split}_mel.h5"

    with h5py.File(h5_path, 'w') as h5_file:
        # Create datasets
        mel_dataset = h5_file.create_dataset(
            'mel_spectrograms',
            shape=(len(metadata), 1, n_mels, expected_mel_frames),
            dtype=np.float32,
            compression='gzip',
            compression_opts=9
        )

        adsr_dataset = h5_file.create_dataset(
            'adsr_parameters',
            shape=(len(metadata), 4),
            dtype=np.float32,
            compression='gzip',
            compression_opts=9
        )

        file_paths_dataset = h5_file.create_dataset(
            'file_paths',
            shape=(len(metadata),),
            dtype=h5py.special_dtype(vlen=str)
        )

        # Store metadata
        h5_file.attrs['unit_sec'] = unit_sec
        h5_file.attrs['sr'] = sr
        h5_file.attrs['n_mels'] = n_mels
        h5_file.attrs['fmin'] = fmin
        h5_file.attrs['fmax'] = fmax
        h5_file.attrs['hop_length'] = hop_length
        h5_file.attrs['n_fft'] = n_fft
        h5_file.attrs['window'] = window
        h5_file.attrs['center'] = center
        h5_file.attrs['power'] = power
        h5_file.attrs['expected_mel_frames'] = expected_mel_frames

        # Process each audio file
        for idx, chunk in enumerate(tqdm(metadata, desc=f"Processing {split} split")):
            try:
                # Load audio
                wav_path = data_dir / split / chunk["file"]
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

                # Convert to mel spectrogram
                mel = mel_converter(wav.to(device))
                # mel = mel.unsqueeze(1)  # Add channel dimension [1, 128, time]

                # Debug: Print mel shape
                if idx == 0:
                    print(f"Original mel shape: {mel.shape}")

                # Debug: Print final mel shape
                if idx == 0:
                    print(f"Final mel shape: {mel.shape}")
                    print(f"Expected shape: (1, {n_mels}, {expected_mel_frames})")

                # Ensure mel is the correct shape and data type
                mel_np = mel.cpu().numpy().astype(np.float32)

                # Verify shape before storing
                if mel_np.shape != (1, n_mels, expected_mel_frames):
                    print(f"Warning: Mel shape mismatch at index {idx}. Expected: (1, {n_mels}, {expected_mel_frames}), Got: {mel_np.shape}")
                    # Reshape if possible
                    if mel_np.shape[0] == 1 and mel_np.shape[1] == n_mels:
                        if mel_np.shape[2] > expected_mel_frames:
                            mel_np = mel_np[:, :, :expected_mel_frames]
                        else:
                            # Pad with zeros
                            padding = expected_mel_frames - mel_np.shape[2]
                            mel_np = np.pad(mel_np, ((0, 0), (0, 0), (0, padding)), mode='constant')

                # Store in HDF5
                mel_dataset[idx] = mel_np

                # Ensure ADSR parameters are correct
                adsr_params = np.array([
                    float(chunk["attack"]),
                    float(chunk["decay"]),
                    float(chunk["sustain"]),
                    float(chunk["release"])
                ], dtype=np.float32)

                adsr_dataset[idx] = adsr_params
                file_paths_dataset[idx] = str(chunk["file"])

            except Exception as e:
                print(f"Error processing file {chunk['file']} at index {idx}: {e}")
                # Fill with zeros or skip
                mel_dataset[idx] = np.zeros((1, n_mels, expected_mel_frames), dtype=np.float32)
                adsr_dataset[idx] = np.zeros(4, dtype=np.float32)
                file_paths_dataset[idx] = str(chunk["file"])

        # Store dataset shapes before closing the file
        mel_shape = mel_dataset.shape
        adsr_shape = adsr_dataset.shape

    print(f"Created HDF5 dataset at {h5_path}")
    print(f"Dataset shape: {mel_shape}")
    print(f"ADSR parameters shape: {adsr_shape}")

    return h5_path

def create_all_splits(
    data_dir: Path,
    output_dir: Path,
    splits: list = ["val", "test", "train"],
    **kwargs
):
    """Create HDF5 datasets for all splits."""

    for split in splits:
        print(f"\nProcessing {split} split...")
        try:
            create_mel_h5_dataset(data_dir, output_dir, split, **kwargs)
            print(f"Successfully processed {split} split")
        except FileNotFoundError:
            print(f"Warning: {split} split not found, skipping...")
        except Exception as e:
            print(f"Error processing {split} split: {e}")
            import traceback
            traceback.print_exc()

        # Remove the break statement to process all splits
        # break

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    data_dir = Path("/home/buffett/dataset/rendered_adsr_unpaired")
    output_dir = Path("/home/buffett/dataset/rendered_adsr_unpaired_h5")

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
        'device': device
    }

    # Create HDF5 datasets for all splits
    create_all_splits(data_dir, output_dir, **mel_params)
