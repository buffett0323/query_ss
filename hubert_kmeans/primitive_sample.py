import librosa
import librosa.display
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Load EDM lead sound file
audio_path = "lead_sound.wav"  # Change to your file path
y, sr = librosa.load(audio_path, sr=None)

# Display waveform
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform of Lead Sound")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# 1. Compute Spectral Flux (Frequency-Domain Onset Detection)
# Compute STFT
stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))

# Compute spectral flux (difference between consecutive frames)
flux = np.sum(np.diff(stft, axis=1) ** 2, axis=0)

# Normalize spectral flux
flux = flux / np.max(flux)

# Plot spectral flux
plt.figure(figsize=(12, 4))
plt.plot(flux, label="Spectral Flux")
plt.title("Spectral Flux Onset Detection")
plt.xlabel("Frame Index")
plt.ylabel("Magnitude Difference")
plt.legend()
plt.show()


# 2. Compute Energy-Based Onset Detection
# Compute short-term energy
frame_length = 2048
hop_length = 512
energy = np.array([
    np.sum(y[i:i+frame_length]**2)
    for i in range(0, len(y) - frame_length, hop_length)
])

# Normalize energy
energy = energy / np.max(energy)

# Plot energy onset detection
plt.figure(figsize=(12, 4))
plt.plot(energy, label="Energy")
plt.title("Energy-Based Onset Detection")
plt.xlabel("Frame Index")
plt.ylabel("Normalized Energy")
plt.legend()
plt.show()


# 3. Phase Deviation
# Compute STFT (get phase)
stft_phase = np.angle(librosa.stft(y, n_fft=2048, hop_length=512))

# Compute phase deviation (difference between consecutive frames)
phase_dev = np.sum(np.diff(stft_phase, axis=1) ** 2, axis=0)

# Normalize phase deviation
phase_dev = phase_dev / np.max(phase_dev)

# 4. Plot
# Plot phase-based onset detection
plt.figure(figsize=(12, 4))
plt.plot(phase_dev, label="Phase Deviation")
plt.title("Phase-Based Onset Detection")
plt.xlabel("Frame Index")
plt.ylabel("Phase Change Magnitude")
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(flux, label="Spectral Flux", alpha=0.7)
plt.plot(energy, label="Energy-Based", alpha=0.7)
plt.plot(phase_dev, label="Phase-Based", alpha=0.7)
plt.title("Comparison of Onset Detection Methods")
plt.xlabel("Frame Index")
plt.ylabel("Normalized Detection Strength")
plt.legend()
plt.show()


from scipy.signal import find_peaks

# Detect peaks in spectral flux
peaks_flux, _ = find_peaks(flux, height=0.2, distance=5)

# Detect peaks in energy
peaks_energy, _ = find_peaks(energy, height=0.2, distance=5)

# Detect peaks in phase deviation
peaks_phase, _ = find_peaks(phase_dev, height=0.2, distance=5)


# Plot results with detected peaks
plt.figure(figsize=(12, 6))
plt.plot(flux, label="Spectral Flux", alpha=0.7)
plt.plot(energy, label="Energy-Based", alpha=0.7)
plt.plot(phase_dev, label="Phase-Based", alpha=0.7)
plt.scatter(peaks_flux, flux[peaks_flux], color="red", label="Onsets (Spectral Flux)")
plt.scatter(peaks_energy, energy[peaks_energy], color="blue", label="Onsets (Energy)")
plt.scatter(peaks_phase, phase_dev[peaks_phase], color="green", label="Onsets (Phase)")
plt.title("Onset Detection Peaks")
plt.xlabel("Frame Index")
plt.ylabel("Detection Strength")
plt.legend()
plt.show()
