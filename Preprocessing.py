# Yason Bedoshvili
# ID No: 40058829
# Part 1

import os
from PIL import Image
import hashlib
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader

# Function that calculates image hash for detecting duplicates
def calculate_hash(image_path):
    with Image.open(image_path) as img:
        img_hash = hashlib.md5(img.tobytes()).hexdigest()
    return img_hash

# Function that finds and removes duplicates
def remove_duplicates(directory):
    image_hashes = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                file_path = os.path.join(root, file)
                img_hash = calculate_hash(file_path)
                if img_hash in image_hashes:
                    print(f"Duplicate found: {file_path} and {image_hashes[img_hash]}")
                    os.remove(file_path)
                else:
                    image_hashes[img_hash] = file_path

# Function that resizes images to 128x128 pixels and convert to PNG format
def verify_and_process_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img = img.resize((128, 128))  # Resize to 128x128 pixels
                        new_file_path = os.path.splitext(file_path)[0] + '.png'
                        img.save(new_file_path, 'PNG')
                        if file_path != new_file_path:
                            os.remove(file_path)  # Remove the original file if format is changed
                except Exception as e:
                    print(f"Error processing image {file_path}: {e}")

# Paths to train and test directories for all categories
dataset_dirs = [
    r"C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\angry\train",
    r"C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\angry\test",
    r"C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\happy\train",
    r"C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\happy\test",
    r"C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\neutral\train",
    r"C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\neutral\test",
    r"C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\focused\train",
    r"C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\focused\test"
]

# Process each directory to remove duplicates and verify images
for directory in dataset_dirs:
    print(f"Processing directory: {directory}")
    remove_duplicates(directory)
    verify_and_process_images(directory)

print("Data preprocessing completed.")
