from diffusers import AudioLDM2Pipeline
import torch

# Load the pre-trained pipeline
repo_id = "cvssp/audioldm2"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Extract the VAE from the pipeline
vae = pipe.vae

# Get the pre-trained encoder
encoder = vae.encoder
# torch.save(encoder.state_dict(), "audio_ldm2_vae_encoder_checkpoint.pth")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
xm = torch.randn(1, 1, 64, 400).to(device)
xm = xm.to(torch.float16)
res = encoder(xm)
print("RESRES:", res.shape)

# Example input
import torch 
from dismix_LDM import Partition
batch_size = 2
Z_s = torch.randn(batch_size, 8, 16, 100)  # Shape: [B, C=8, D=16, T=100]

# Partition configuration
patch_size = 4  # 4 consecutive time steps per patch
num_patches = 25  # Total patches along the time axis
dim = 512  # Feature dimension after flattening

# Initialize and forward
partition = Partition(patch_size=patch_size, dim=dim, num_patches=num_patches)
output = partition(Z_s)

print(output.shape)  # Output: torch.Size([2, 25, 512])
