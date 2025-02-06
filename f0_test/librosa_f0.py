import torchcrepe
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio
path = "/mnt/gestalt/database/beatport/audio/audio/electro-big-room/51c00cca-0453-4b31-a4ce-e13ceec6b9b9.mp3"
y, sr = librosa.load(path, sr=44100)

start = 44100 * 15
end = 44100 * 20
y = y[start:end]

# Extract F0 using pYIN
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=2000, sr=sr)

# Convert NaNs (unvoiced frames) to 0 for cleaner visualization
f0 = np.nan_to_num(f0)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max)

# Time axis for F0
times = librosa.times_like(f0, sr=sr, hop_length=512)

# Plot Spectrogram & F0 Overlay
plt.figure(figsize=(10, 6))
librosa.display.specshow(D, sr=sr, hop_length=512, x_axis="time", y_axis="log")
plt.plot(times, f0, color="green", linewidth=2, label="Estimated F0 (pYIN)")
plt.colorbar(label="dB")
plt.title("Spectrogram with pYIN F0 Overlay")
plt.legend()
plt.savefig('results/librosa_f0.png')
