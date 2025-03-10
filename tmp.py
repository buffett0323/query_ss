import numpy as np
print(np.__version__)

# import os
# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm

# # Simple Model
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.fc = nn.Linear(10, 1)

#     def forward(self, x):
#         return self.fc(x)

# # Training Function
# if __name__ == "__main__":
#     device = torch.device("cuda:0")

#     # Create model, move to GPU, and wrap with DDP
#     model = SimpleModel().to(device)
    
#     dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
#     dataloader = DataLoader(dataset, batch_size=16)

#     # Loss and optimizer
#     criterion = nn.MSELoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.01)

#     # Training loop
#     for epoch in tqdm(range(1000)):
#         for batch in dataloader:
#             inputs, targets = batch
#             inputs, targets = inputs.to(device), targets.to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()

#         print(f"Epoch {epoch}, Loss: {loss.item()}")