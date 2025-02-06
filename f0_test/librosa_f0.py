import torchcrepe
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio
path = "/mnt/gestalt/database/beatport/audio/audio/electro-big-room/51c00cca-0453-4b31-a4ce-e13ceec6b9b9.mp3"
y, sr = librosa.load(path, sr=44100)

start = 0 # 44100 * 15
end = 44100 * 3
y = y[start:end]

# Extract F0 using pYIN
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=2000, sr=sr)

# Convert NaNs (unvoiced frames) to 0
f0 = np.nan_to_num(f0)

# Convert F0 to MIDI Notes (ignore zero values)
f0_nonzero = np.where(f0 > 0, f0, np.nan)
midi_notes = librosa.hz_to_midi(f0_nonzero)

# Create time axis
times = librosa.times_like(f0, sr=sr, hop_length=512)

# Plot MIDI Notes over Time
plt.figure(figsize=(10, 5))
plt.scatter(times, midi_notes, s=5, c="red", alpha=0.7, label="Estimated MIDI Notes")
plt.ylim(21, 108)  # MIDI range (A0 to C8)
plt.xlabel("Time (seconds)")
plt.ylabel("MIDI Note Number")
plt.title("MIDI Notes Over Time")
plt.legend()
plt.grid()


# # Time axis for F0
# times = librosa.times_like(f0, sr=sr, hop_length=512)

# # Plot Spectrogram & F0 Overlay
# plt.figure(figsize=(10, 6))
# librosa.display.specshow(D, sr=sr, hop_length=512, x_axis="time", y_axis="log")
# plt.plot(times, f0, color="green", linewidth=2, label="Estimated F0 (pYIN)")
# plt.colorbar(label="dB")
# plt.title("Spectrogram with pYIN F0 Overlay")
# plt.legend()
plt.savefig('results/librosa_f0.png')
