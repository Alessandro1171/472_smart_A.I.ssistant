import torch
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all optimization algortihms, SDG, Adam, etc.
import torchvision
import torchvision.transforms as transforms # Transformations we can perform on our datasets
from torch.utils.data import DataLoader # Gives easier dataset management and creates mini batches
from CustomDataset import HappyDataset
from torchvision.utils import save_image
import os

# Target numbers: 1 == Neutral, 2 == Engaged, 3 == Angry, 4 == Happy

# Data cleaning for training dataset
my_training_transformation = transforms.Compose([
    transforms.ToPILImage(), # format to allow transformations to work
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale since most images are already black and white
    transforms.Resize((128, 128)), # Make images all the same size
    transforms.RandomHorizontalFlip(), # Adds variety
    transforms.RandomRotation(degrees=15), # Adds robustness
    transforms.ToTensor(), # Transforms the image back to a readable format
    #transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize for grayscale
])

# Data cleaning for testing dataset
my_testing_transformation = transforms.Compose([
    transforms.ToPILImage(),  # Convert to PIL Image (if needed)
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((128, 128)),  # Resize to a common size
    transforms.ToTensor(),  # Convert to tensor
    #transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale
])

# Define output directories for training and testing sets
train_output_dir = r"C:\Users\henar\PycharmProjects\COMP471\venv\Smart A.I.ssistant\train\happy_output"
test_output_dir = r"C:\Users\henar\PycharmProjects\COMP471\venv\Smart A.I.ssistant\test\happy_output"

# load data
happy_train_set = HappyDataset(csv_file="happy_training.csv", root_dir=r"C:\Users\henar\PycharmProjects\COMP471\venv\Smart A.I.ssistant\train\happy",
                       transform= my_training_transformation)
happy_test_set = HappyDataset(csv_file="happy_testing.csv", root_dir=r"C:\Users\henar\PycharmProjects\COMP471\venv\Smart A.I.ssistant\test\happy",
                       transform= my_testing_transformation)

# Renames all images with ID numbers and converts them to .png files so data is not lost when manipulated
img_id = 0
for img, label in happy_train_set:
    save_image(img, os.path.join(train_output_dir, f'happy_training_{img_id}.png'))
    img_id += 1

img_id = 0
for img, label in happy_test_set:
    save_image(img, os.path.join(test_output_dir, f'happy_testing_{img_id}.png'))
    img_id += 1
