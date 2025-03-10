## Fuck off## Fuck off
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Simple Model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Training Function
def train(rank, world_size):
    # Initialize process group
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12359'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Create model, move to GPU, and wrap with DDP
    model = SimpleModel().to(device)
    model = DDP(model, device_ids=[rank])

    # Create synthetic dataset
    dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in tqdm(range(3)):
        sampler.set_epoch(epoch)  # Ensures randomness across epochs
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    # Clean up
    dist.destroy_process_group()

# Multi-GPU Entry Point
if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # Get number of available GPUs
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
