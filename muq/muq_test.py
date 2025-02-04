import torch, librosa
from muq import MuQ

device = 'cuda:0'
path = '/mnt/gestalt/database/beatport/audio/audio/electro-house'
wav, sr = librosa.load(f"{path}/76817fd1-6742-4e23-aa6c-a9728a2fc493.mp3", sr = 44100)
wavs = torch.tensor(wav).unsqueeze(0).to(device) 

# This will automatically fetch the checkpoint from huggingface
muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
muq = muq.to(device).eval()

with torch.no_grad():
    output = muq(wavs, output_hidden_states=True)

print('Total number of layers: ', len(output.hidden_states))
print('Feature shape: ', output.last_hidden_state.shape)
