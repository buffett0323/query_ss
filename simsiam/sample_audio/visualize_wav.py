import matplotlib.pyplot as plt
import numpy as np
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize audio waveform from .npy file")
    parser.add_argument("--npy_file", type=str, required=True, help="Path to the .npy file")
    parser.add_argument("--output_fig", type=str, default="waveform.png", help="Path to save the output figure")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate of the audio")
    
    args = parser.parse_args()
    waveform = np.load(args.npy_file)
    sample_rate = args.sample_rate  # Based on the context files showing 16000Hz sample rate

    # Ensure waveform is 2D with shape [channels, num_samples]
    if len(waveform.shape) == 1:
        waveform = waveform[np.newaxis, :]

    # Convert x-axis to seconds
    num_samples = waveform.shape[1]
    duration = num_samples / sample_rate
    time_axis = np.linspace(0, duration, num_samples)

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, waveform[0], label="Channel 1")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output_fig)
    plt.close()