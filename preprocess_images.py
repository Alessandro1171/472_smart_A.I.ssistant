# Yason Bedoshvili
# ID No: 40058829
# Part 1

import os
from PIL import Image


def preprocess_images(input_dir, output_dir, target_size=(128, 128)):
    """
    Preprocess images by resizing them to the target size, converting them to grayscale,
    and saving them in PNG format.
    """

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through each file in the input directory
    for filename in os.listdir(input_dir):
        # Process only image files with common extensions
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            # Construct the full path to the image
            img_path = os.path.join(input_dir, filename)

            # Open the image using PIL
            img = Image.open(img_path)

            # Convert the image to grayscale
            img = img.convert('L')

            # Resize the image to the target size
            img = img.resize(target_size)

            # Construct the output path with PNG extension
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')

            # Save the processed image in PNG format
            img.save(output_path, 'PNG')

            # Print a message indicating successful processing
            print(f"Processed and saved: {output_path}")


# Defining directories
input_directory = r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\OurPhotos'
output_directory = r'C:\Users\yason\OneDrive\Documents\summer_2024\COMP_472\ProcessedPhotos'

# Preprocess images in the input directory and save them in the output directory
preprocess_images(input_directory, output_directory)
