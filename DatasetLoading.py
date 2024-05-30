# Yason Bedoshvili
# ID No: 40058829
# Part 1

import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader

# Custom dataset class for loading images
class FacialExpressionDataset(Dataset):
    def __init__(self, dataset_paths, transform=None):
        self.dataset_paths = dataset_paths
        self.transform = transform
        self.image_files = []
        for path in dataset_paths:
            self.image_files.extend([(os.path.join(path, f), path.split(os.sep)[-2]) for f in os.listdir(path) if f.endswith(".png")])
        self.classes = sorted(list(set([label for _, label in self.image_files])))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name, label = self.image_files[idx]
        image = Image.open(img_name)  # Images are already grayscale
        if self.transform:
            image = self.transform(image)
        else:
            image_array = np.array(image) / 255.0  # Normalization of pixel values
            image = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)  # Addition of channel dimension
        label_idx = self.classes.index(label)
        return image, label_idx

# Defining the augmentation transformations
augmentation_transforms = transforms.Compose([
    transforms.RandomRotation(10),  # Rotate by -10 to +10 degrees
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomResizedCrop(128, scale=(0.9, 1.1)),  # Random crop and resize
    transforms.ToTensor()  # Convert image to PyTorch tensor
])

# Defining the normalization transformation
normalization_transform = transforms.Compose([
    transforms.ToTensor()  # Convertion of image to PyTorch tensor
])

# Paths to datasets
train_dataset_paths = [
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\angry\train',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\happy\train',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\neutral\train',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\focused\train'
]

test_dataset_paths = [
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\angry\test',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\happy\test',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\neutral\test',
    r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\focused\test'
]

# Loading the datasets with augmentation for training and without for testing
augmented_train_dataset = FacialExpressionDataset(train_dataset_paths, transform=augmentation_transforms)
normalized_test_dataset = FacialExpressionDataset(test_dataset_paths, transform=normalization_transform)

# Example usage with DataLoader
batch_size = 32
augmented_train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True)
normalized_test_loader = DataLoader(normalized_test_dataset, batch_size=batch_size, shuffle=False)

# Iteration through the dataset (example)
for images, labels in augmented_train_loader:
    print(images.size())  # Should print torch.Size([batch_size, 1, 128, 128])
    print(labels.size())  # Should print torch.Size([batch_size])
    # Training code should be here

for images, labels in normalized_test_loader:
    print(images.size())  # Should print torch.Size([batch_size, 1, 128, 128])
    print(labels.size())  # Should print torch.Size([batch_size])
    # Testing/evaluation code should be here
