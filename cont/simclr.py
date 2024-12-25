import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, TimeStretch, PitchShift, AdditiveNoise
from tqdm import tqdm


# SimCLR Loss: NT-Xent Loss
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: Latent embeddings from first augmented view [batch_size, embed_dim].
            z_j: Latent embeddings from second augmented view [batch_size, embed_dim].
        Returns:
            Loss: Contrastive loss.
        """
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # Combine embeddings
        z = F.normalize(z, dim=1)  # Normalize

        sim_matrix = torch.mm(z, z.T) / self.temperature  # Cosine similarity matrix
        sim_matrix.fill_diagonal_(-float('inf'))  # Remove self-similarity

        # Positive pairs are on the diagonal of upper-right and lower-left blocks
        positive_samples = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0)
        positive_samples = positive_samples + batch_size * (positive_samples < batch_size).long()

        positive_scores = sim_matrix[torch.arange(2 * batch_size), positive_samples]
        exp_scores = torch.exp(sim_matrix)

        # Compute NT-Xent loss
        loss = -torch.log(torch.exp(positive_scores) / exp_scores.sum(dim=1))
        return loss.mean()



# Timbre Encoder with Projection Head
class TimbreSimCLR(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, projection_dim=64):
        super(TimbreSimCLR, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(input_dim // 4 * input_dim // 4 * 64, hidden_dim),
            nn.ReLU()
        )
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return features, projections


# Dataset for SimCLR
class SimCLRDataset(Dataset):
    def __init__(self, audio_files, transforms):
        self.audio_files = audio_files
        self.transforms = transforms

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio = self.audio_files[idx]
        audio_1 = self.transforms(audio)
        audio_2 = self.transforms(audio)
        return audio_1, audio_2

# Data Augmentation
def audio_augmentations():
    return nn.Sequential(
        MelSpectrogram(),
        TimeStretch(),
        PitchShift(),
        AdditiveNoise()
    )

# Training Loop
def train_simclr(encoder, dataset, epochs=10, batch_size=32, learning_rate=1e-3):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    loss_fn = NTXentLoss()

    for epoch in range(epochs):
        encoder.train()
        total_loss = 0
        for audio_1, audio_2 in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            audio_1, audio_2 = audio_1.cuda(), audio_2.cuda()
            _, z_i = encoder(audio_1)
            _, z_j = encoder(audio_2)

            loss = loss_fn(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

# Example Usage
encoder = TimbreSimCLR().cuda()
transforms = audio_augmentations()
# Assuming `audio_files` is a preloaded array of audio tensors.
dataset = SimCLRDataset(audio_files, transforms)
train_simclr(encoder, dataset)
