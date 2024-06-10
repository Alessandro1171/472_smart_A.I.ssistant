import os
import random
import shutil


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


# Base paths for your datasets
base_paths = {
    "angry": r"C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AI_Part2\angry",
    "happy": r"C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AI_Part2\happy",
    "neutral": r"C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AI_Part2\neutral",
    "focused": r"C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\AI_Part2\focused"
}

split_dataset(base_paths)
