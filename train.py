# This file is deprecated and no longer in use.
# Please refer to the newer files for the latest implementation.

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch.nn as nn
from PIL import Image
import numpy as np
import os
from model import SimpleCNN



class FacialExpressionDataset(Dataset):
    def __init__(self, dataset_paths, transform=None):
        self.dataset_paths = dataset_paths
        self.transform = transform
        self.image_files = []
        for path in dataset_paths:
            self.image_files.extend(
                [(os.path.join(path, f), path.split(os.sep)[-2]) for f in os.listdir(path) if f.endswith(".png")])
        self.classes = sorted(list(set([label for _, label in self.image_files])))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name, label = self.image_files[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        else:
            image_array = np.array(image) / 255.0
            image = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)
        label_idx = self.classes.index(label)
        return image, label_idx


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Paths to datasets
train_dataset_paths = [
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AI_Part2\angry\train',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AI_Part2\happy\train',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AI_Part2\neutral\train',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AI_Part2\focused\train'
]

val_dataset_paths = [
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AI_Part2\angry\val',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AI_Part2\happy\val',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AI_Part2\neutral\val',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AI_Part2\focused\val'
]

train_dataset = FacialExpressionDataset(train_dataset_paths, transform=transform)
val_dataset = FacialExpressionDataset(val_dataset_paths, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
patience = 3
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(10):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

    val_loss = val_loss / len(val_loader.dataset)

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break
