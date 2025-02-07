from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import pretty_midi
import matplotlib.pyplot as plt


path = "/mnt/gestalt/home/ddmanddman/beatport_analyze/htdemucs/51c00cca-0453-4b31-a4ce-e13ceec6b9b9/other.wav"
model_output, midi_data, note_events = predict(path)
print(model_output['contour'].shape)
# print(note_events.shape)


# Get the piano roll
piano_roll = midi_data.get_piano_roll()

# Plot the piano roll
plt.figure(figsize=(10, 6))
plt.imshow(piano_roll[65:75, :100], aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
plt.xlabel("Time (frames)")
plt.ylabel("MIDI Pitch")
plt.title("Piano Roll Visualization")
plt.colorbar(label="Velocity")
plt.savefig('results/piano_roll.png')
# plt.show()
