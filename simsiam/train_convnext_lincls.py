import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Step 1: Define the SimSiam backbone (we assume the pre-trained model is loaded)
class SimSiamBackbone(nn.Module):
    def __init__(self):
        super(SimSiamBackbone, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=False)
        self.model.fc = nn.Identity()  # Removing the final fully connected layer
    
    def forward(self, x):
        return self.model(x)  # Output feature vector from ResNet50

# Step 2: Load ImageNet dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Example: Using the ImageNet dataset (you can replace this with your custom dataset)
train_dataset = torchvision.datasets.ImageNet(root='path_to_imagenet', split='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Step 3: Define the Linear Classifier
class LinearClassifier(nn.Module):
    def __init__(self, in_features, num_classes=1000):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)  # 1000 classes for ImageNet
    
    def forward(self, x):
        return self.fc(x)

# Step 4: Load the pretrained SimSiam model (assuming it's already trained)
backbone = SimSiamBackbone()
backbone.load_state_dict(torch.load('simsiam_backbone.pth'))  # Load pre-trained weights
backbone.eval()  # Set backbone to evaluation mode
backbone.requires_grad_(False)  # Freeze the backbone (no gradients)

# Step 5: Create the classifier on top of the frozen backbone
classifier = LinearClassifier(in_features=2048, num_classes=1000)  # 2048 is the output size of ResNet50

# Step 6: Define optimizer and loss function
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Step 7: Training loop for the linear classifier
def train_linear_classifier():
    classifier.train()  # Set classifier to training mode
    for images, labels in train_loader:
        optimizer.zero_grad()
        features = backbone(images)  # Extract features from the frozen backbone
        outputs = classifier(features)  # Get predictions from the linear classifier
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()
        optimizer.step()  # Update classifier weights

# Train the classifier
train_linear_classifier()

# Step 8: Evaluate performance
def evaluate_classifier():
    classifier.eval()  # Set classifier to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in train_loader:
            features = backbone(images)  # Extract features
            outputs = classifier(features)  # Get predictions
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            correct += (predicted == labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Total samples
    accuracy = correct / total  # Calculate accuracy
    print(f'Accuracy: {accuracy * 100:.2f}%')

# Evaluate the classifier
evaluate_classifier()