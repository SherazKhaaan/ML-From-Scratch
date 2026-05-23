import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1) Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Transforms convert images to PyTorch Tensors and Normalise them
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 3) DataLoaders for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 128, shuffle = False)

# 4) Model Architecture (784 - 256 - 128 - 10)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),           # Flatten 28x28 images to 784-dimensional vectors
            nn.Linear(784, 256),    # Input Layer -> Hidden Layer 1
            nn.ReLU(),              
            nn.Linear(256, 128),    # Hidden Layer 1 -> Hidden Layer 2 
            nn.ReLU(),
            nn.Linear(128, 10)      # Hidden Layer 2 -> Output layer with 10 classes (digits 0-9)
        )


    def forward(self, x):
        return self.network(x)
    

# 5) Instantiate model, define loss function and optimizer
model = MLP().to(device)

criterion = nn.CrossEntropyLoss() # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6) Training Loop
epochs = 5 
print(f"Training on {device}...\n")

for epoch in range(epochs):
    model.train() 
    total_loss = 0 

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device) # Move data to GPU if available

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimisation 
        optimizer.zero_grad() # Clear old gradients
        loss.backward()       # Autograd engine computes all dW and dB derivatives 
        optimizer.step()      # Adam adjusts the weights based on those gradients

        # Total loss = sum of batch loss * num samples in that batch
        total_loss += loss.item() * images.size(0) 

    
    # Evaluate at the end of each epoch 
    model.eval()
    correct, total = 0, 0

    with torch.no_grad(): # Disables gradient calculation for evaluation (saves memory and computations)
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) # Pick the class with the highest score as prediction
            total += labels.size(0)
            correct += (predicted == labels).sum().item() # Count how many predictions were correct

    epoch_loss = total_loss / len(train_loader.dataset) # Average loss per sample
    epoch_accuracy = (correct / total) * 100 # Accuracy percentage
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.2f}%")

        