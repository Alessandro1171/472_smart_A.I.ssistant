import os
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F


# Dataset Split Function
def split_dataset(base_paths, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    for category, path in base_paths.items():
        train_path = os.path.join(path, "train")
        val_path = os.path.join(path, "val")
        test_path = os.path.join(path, "test")

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        image_files = [f for f in os.listdir(path) if f.endswith(".jpg") or f.endswith(".png")]
        random.shuffle(image_files)

        total_images = len(image_files)
        train_end = int(train_ratio * total_images)
        val_end = train_end + int(val_ratio * total_images)

        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]

        for file in train_files:
            shutil.move(os.path.join(path, file), os.path.join(train_path, file))
        for file in val_files:
            shutil.move(os.path.join(path, file), os.path.join(val_path, file))
        for file in test_files:
            shutil.move(os.path.join(path, file), os.path.join(test_path, file))
    print("Dataset split completed.")


# CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128 * 26 * 26, 256)
        self.fc2 = nn.Linear(256, 4)  # 4 output classes: angry, happy, neutral, focused

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Custom Dataset Class
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


# Paths to datasets
base_paths = {
    "angry": r"C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\angry",
    "happy": r"C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\happy",
    "neutral": r"C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\neutral",
    "focused": r"C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\focused"
}

split_dataset(base_paths)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset_paths = [
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\angry\train',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\happy\train',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\neutral\train',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\focused\train'
]

val_dataset_paths = [
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\angry\val',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\happy\val',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\neutral\val',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\focused\val'
]

test_dataset_paths = [
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\angry\test',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\happy\test',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\neutral\test',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\focused\test'
]

train_dataset = FacialExpressionDataset(train_dataset_paths, transform=transform)
val_dataset = FacialExpressionDataset(val_dataset_paths, transform=transform)
test_dataset = FacialExpressionDataset(test_dataset_paths, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

# Evaluate the Model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes,
            yticklabels=test_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
report = classification_report(all_labels, all_preds, target_names=test_dataset.classes, output_dict=True)
print(report)

# Save the report
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('classification_report.csv')
