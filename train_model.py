"""
train_model.py
----------------
Trains the brain tumor classification CNN and saves the weights +
class names into model.pth, ready for the Flask backend to load.

Expects the standard folder layout:
    Training/<class_name>/*.jpg
    Testing/<class_name>/*.jpg

Usage:
    python train_model.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_def import CNN

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    root="Training",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root="Testing",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

class_names = train_dataset.classes
print("Detected classes:", class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

model = CNN(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
model.train()

for epoch in range(epochs):
    running_loss = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(epoch + 1, running_loss)


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        test_outputs = model(x_batch)
        _, predicted = torch.max(test_outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    accuracy = correct / total * 100

print("Accuracy Score Is: ", accuracy, "%")

# ---------------- Save model for deployment ----------------
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
    "accuracy": accuracy
}, "model.pth")

print("\nSaved trained model to model.pth")
print("You can now run: python app.py")
