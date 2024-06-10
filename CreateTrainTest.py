# Yason Bedoshvili
# ID No: 40058829
# Part 1

import os
import shutil
import random

# Paths of datasets that are not divided on train and test
base_paths = {
    "angry": r"C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\angry",
    "happy": r"C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\happy",
    "neutral": r"C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AllFaces_updated\neutral"
}

# Test and train directories will be created below
for category, path in base_paths.items():
    # Define train and test paths
    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "test")

    # Create directories if they don't exist
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # List all images in the category directory
    image_files = [f for f in os.listdir(path) if f.endswith(".jpg") or f.endswith(".png")]

    # Shuffle the list of files for random distribution
    random.shuffle(image_files)

    # 80% train and 20% test
    split_index = int(0.8 * len(image_files))

    # Move files to train and test directories based on the split index
    for i, file in enumerate(image_files):
        src_file = os.path.join(path, file)
        if i < split_index:
            dest_dir = train_path
        else:
            dest_dir = test_path
        shutil.move(src_file, os.path.join(dest_dir, file))

print("Train/Test split completed.")
