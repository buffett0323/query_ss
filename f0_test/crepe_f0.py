import torchcrepe
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio
path = "/mnt/gestalt/database/beatport/audio/audio/electro-big-room/51c00cca-0453-4b31-a4ce-e13ceec6b9b9.mp3"
audio, sr = torchaudio.load(path)

if audio.shape[0] > 1:
    audio = audio.mean(dim=0, keepdim=False)  # Convert to mono



target_sample_rate = 16000
if sr != target_sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
    audio = resampler(audio)



start = target_sample_rate * 15
end = target_sample_rate * 20
audio = audio[start:end]

# Here we'll use a 5 millisecond hop length
hop_length = 1024 # int(sr / 200.)

# Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
# This would be a reasonable range for speech
fmin = 50
fmax = 550

# Select a model capacity--one of "tiny" or "full"
model = 'tiny'
device = 'cuda:2'
batch_size = 1 #2048

# Compute pitch using first gpu
audio = audio.unsqueeze(0)


f0 = torchcrepe.predict(
    audio,
    target_sample_rate,
    hop_length,
    fmin=fmin,
    fmax=fmax,
    model="full",
    batch_size=batch_size,
).squeeze(0).numpy()  # Convert to NumPy

f0[np.isnan(f0)] = 0


# Compute the spectrogram
y = audio.squeeze().numpy()
D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length)), ref=np.max)

# Plot Spectrogram & F0
plt.figure(figsize=(10, 6))
librosa.display.specshow(D, sr=sr, hop_length=512, x_axis="time", y_axis="log")
plt.plot(np.linspace(0, len(y) / sr, num=len(f0)), f0, color="green", label="F0 Estimate")
plt.colorbar(label="dB")
plt.title("Spectrogram with F0 Overlay")
plt.legend()
plt.savefig('results/crepe.png')
