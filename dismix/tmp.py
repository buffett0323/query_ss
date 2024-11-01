import torch
import torch.nn.functional as F

# Step 1: Generate a tensor of 32 random labels from 0 to 51
labels = torch.randint(0, 52, (32,))
print(labels.shape)

# Step 2: Apply one-hot encoding to convert each label to a 52-length vector
one_hot_encoded = F.one_hot(labels, num_classes=52)

print("Labels:", labels)
print("One-Hot Encoded Tensor Shape:", one_hot_encoded.shape)  # Should output (32, 52)
print(one_hot_encoded)  # Display the one-hot encoded result

