import numpy as np
import soundfile as sf
import argparse
from pydub import AudioSegment

def npy_to_mp3(npy_file, output_mp3, sample_rate=44100):
    """
    Converts a .npy file containing an audio waveform into an MP3 file.

    Args:
        npy_file (str): Path to the .npy file.
        output_mp3 (str): Path to save the MP3 file.
        sample_rate (int, optional): Sampling rate of the audio. Default is 44100 Hz.
    """
    # Load the numpy array
    audio_data = np.load(npy_file)

    # Ensure it's a float32 format (soundfile expects this format)
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Save as a temporary WAV file
    temp_wav = "temp_audio.wav"
    sf.write(temp_wav, audio_data, samplerate=sample_rate)

    # Convert WAV to MP3
    audio = AudioSegment.from_wav(temp_wav)
    audio.export(output_mp3, format="mp3")

    print(f"MP3 file saved as: {output_mp3}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npy audio to .mp3")
    parser.add_argument("npy_file", type=str, help="Path to the .npy file")
    parser.add_argument("--output_mp3", type=str, help="Path to the output MP3 file")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate (default: 44100 Hz)")

    args = parser.parse_args()
    npy_to_mp3(args.npy_file, args.output_mp3, args.sample_rate)
