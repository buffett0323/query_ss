import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Parameters
TARGET_DURATION = 5 # seconds
SR = 44100  # sampling rate
SELECTION_METHOD = "energy"  # Options: "energy", "spectral_complexity", "onset_density", "combined"

# Paths
input_path = "/mnt/gestalt/home/buffett/beatport_analyze/chorus_audio_44100_wav_other_new"
output_path = f"/mnt/gestalt/home/buffett/beatport_analyze/chorus_audio_44100_wav_other_new_{TARGET_DURATION}secs_{SELECTION_METHOD}"
os.makedirs(output_path, exist_ok=True)


def find_best_segment_energy(audio, target_length, hop_length=SR//4):
    """Find segment with highest RMS energy."""
    if len(audio) <= target_length:
        return 0

    best_start = 0
    best_energy = 0

    for start in range(0, len(audio) - target_length + 1, hop_length):
        segment = audio[start:start + target_length]
        energy = np.sqrt(np.mean(segment**2))  # RMS energy
        if energy > best_energy:
            best_energy = energy
            best_start = start

    return best_start


def find_best_segment_energy_vectorized(audio, target_length, hop_length=SR//4):
    """Vectorized version using sliding window."""
    if len(audio) <= target_length:
        return 0

    starts = list(range(0, len(audio) - target_length + 1, hop_length))
    segments = np.array([audio[start:start + target_length] for start in starts])

    # Compute all energies at once
    energies = np.sqrt(np.mean(segments**2, axis=1))

    # Find best segment
    best_idx = np.argmax(energies)
    return starts[best_idx]


def find_best_segment_energy_sliding_window(audio, target_length, hop_length=SR//4):
    """Most efficient version using sliding window view."""
    if len(audio) <= target_length:
        return 0

    from numpy.lib.stride_tricks import sliding_window_view

    # Create sliding window view (memory efficient)
    windowed = sliding_window_view(audio, target_length)

    # Sample at hop_length intervals
    sampled_windows = windowed[::hop_length]

    # Compute all energies at once
    energies = np.sqrt(np.mean(sampled_windows**2, axis=1))

    # Find best segment
    best_idx = np.argmax(energies)
    return best_idx * hop_length


def find_best_segment_spectral_complexity(audio, sr, target_length, hop_length=SR//4):
    """Find segment with highest spectral complexity."""
    if len(audio) <= target_length:
        return 0

    best_start = 0
    best_complexity = 0

    for start in range(0, len(audio) - target_length + 1, hop_length):
        segment = audio[start:start + target_length]

        # Compute spectral features
        stft = librosa.stft(segment)
        spectral_centroids = librosa.feature.spectral_centroid(S=np.abs(stft), sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(S=np.abs(stft), sr=sr)[0]
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)

        # Calculate complexity as variance in spectral features
        complexity = (np.var(spectral_centroids) +
                     np.var(spectral_rolloff) +
                     np.mean(np.var(mfccs, axis=1)))

        if complexity > best_complexity:
            best_complexity = complexity
            best_start = start

    return best_start


def find_best_segment_onset_density(audio, sr, target_length, hop_length=SR//4):
    """Find segment with highest onset density (most musical events)."""
    if len(audio) <= target_length:
        return 0

    best_start = 0
    best_density = 0

    for start in range(0, len(audio) - target_length + 1, hop_length):
        segment = audio[start:start + target_length]

        # Detect onsets
        onset_frames = librosa.onset.onset_detect(y=segment, sr=sr, units='time')
        density = len(onset_frames) / TARGET_DURATION  # onsets per second

        if density > best_density:
            best_density = density
            best_start = start

    return best_start


def find_best_segment_combined(audio, sr, target_length, hop_length=SR//4):
    """Combine multiple methods with weighted scoring."""
    if len(audio) <= target_length:
        return 0

    best_start = 0
    best_score = 0

    for start in range(0, len(audio) - target_length + 1, hop_length):
        segment = audio[start:start + target_length]

        # Energy score (normalized)
        energy = np.sqrt(np.mean(segment**2))

        # Spectral complexity score
        stft = librosa.stft(segment)
        spectral_centroids = librosa.feature.spectral_centroid(S=np.abs(stft), sr=sr)[0]
        complexity = np.var(spectral_centroids)

        # Onset density score
        onset_frames = librosa.onset.onset_detect(y=segment, sr=sr, units='time')
        density = len(onset_frames) / TARGET_DURATION

        # Combined score (weighted)
        score = (0.4 * energy + 0.3 * complexity + 0.3 * density)

        if score > best_score:
            best_score = score
            best_start = start

    return best_start


def select_best_segment(audio, sr, target_length, method="energy"):
    """Select the best segment based on the specified method."""
    if method == "energy":
        return find_best_segment_energy_sliding_window(audio, target_length)
        # return find_best_segment_energy(audio, target_length)
    elif method == "spectral_complexity":
        return find_best_segment_spectral_complexity(audio, sr, target_length)
    elif method == "onset_density":
        return find_best_segment_onset_density(audio, sr, target_length)
    elif method == "combined":
        return find_best_segment_combined(audio, sr, target_length)
    else:
        return 0  # Default to first segment


def process_single_file(file, input_path, output_path, target_duration=TARGET_DURATION, sr=SR, method=SELECTION_METHOD):
    """
    Process a single audio file to extract the most informative N seconds.

    Args:
        file: Filename to process
        input_path: Input directory path
        output_path: Output directory path
        target_duration: Target duration in seconds
        sr: Sampling rate
        method: Selection method for best segment

    Returns:
        tuple: (success: bool, filename: str, error_message: str or None)
    """
    try:
        input_file = os.path.join(input_path, file)
        output_file = os.path.join(output_path, file)

        # Load audio file
        audio, sr = librosa.load(input_file, sr=sr, mono=True)

        # Calculate target length in samples
        target_length = int(target_duration * sr)

        if len(audio) > target_length:
            # Find the best segment using the specified method
            best_start = select_best_segment(audio, sr, target_length, method)
            audio = audio[best_start:best_start + target_length]
        elif len(audio) < target_length:
            # If shorter than target duration, pad with zeros
            padding = np.zeros(target_length - len(audio))
            audio = np.concatenate([audio, padding])

        # Save the processed audio
        sf.write(output_file, audio, sr, subtype='PCM_16')
        return (True, file, None)

    except Exception as e:
        return (False, file, str(e))


def main():
    # Get list of all WAV files
    wav_files = [f for f in os.listdir(input_path) if f.endswith('.wav')]#[:100]

    if not wav_files:
        print("No WAV files found in the input directory.")
        return

    print(f"Using selection method: {SELECTION_METHOD}")
    print("Available methods: energy, spectral_complexity, onset_density, combined")

    # Use multiprocessing to process files in parallel
    num_processes = min(mp.cpu_count(), 16)
    print(f"Processing {len(wav_files)} files using {num_processes} processes...")

    # Create partial function with fixed arguments
    process_func = partial(
        process_single_file,
        input_path=input_path,
        output_path=output_path,
        target_duration=TARGET_DURATION,
        sr=SR,
        method=SELECTION_METHOD
    )

    # Process files with multiprocessing
    success_count = 0
    failed_files = []

    with mp.Pool(processes=num_processes) as pool:
        # Use imap for progress tracking
        results = []
        for result in tqdm(pool.imap(process_func, wav_files), total=len(wav_files), desc="Processing files"):
            results.append(result)

            success, filename, error = result
            if success:
                success_count += 1
            else:
                failed_files.append((filename, error))
                print(f"Failed to process {filename}: {error}")

    # Print summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{len(wav_files)} files")
    print(f"Processing rate: {success_count/len(wav_files)*100:.1f}%")
    print(f"Selection method used: {SELECTION_METHOD}")

    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for filename, error in failed_files:
            print(f"- {filename}: {error}")

    print(f"\nOutput files saved to: {output_path}")

if __name__ == "__main__":
    main()
