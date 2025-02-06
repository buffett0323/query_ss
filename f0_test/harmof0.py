import harmof0
import torchaudio
import matplotlib.pyplot as plt



pit = harmof0.PitchTracker()
path = "/mnt/gestalt/database/beatport/audio/audio/electro-big-room/51c00cca-0453-4b31-a4ce-e13ceec6b9b9.mp3"
audio, sr = torchaudio.load(path)


if audio.shape[0] > 1:
    audio = audio.mean(dim=0, keepdim=False)  # Convert to mono

start = 44100 * 15
end = 44100 * 20
audio = audio[start:end]



time, freq, activation, activation_map = pit.pred(audio, sr)
print("Time:", time.shape)
print("Freq:", freq.shape)
print("Activation:", activation.shape)
print("Activation map:", activation_map.shape)

plt.imsave("results/harmof0.png", activation_map.T[::-1])
