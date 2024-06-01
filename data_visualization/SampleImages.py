import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import glob
import random
from PIL import Image


class SampleImages:
    """
    Displays a grid of sample images for a single class
    """

    def __init__(self, path, rows=5, column=3):
        self.rows = rows
        self.column = column
        self.path = path + "/*.png"

    def display_grid(self):
        """
        Presents a collection of 15 sample images in a 5 × 3 (rows ×
        columns) grid for each class. Each image has its pixel intensity histogram next to it.
        """
        image_paths_angry = glob.glob(self.path)
        colors = ["green", "blue", "black", "red"]
        fig, axes = plt.subplots(self.rows, (self.column * 2), figsize=(15, 10))
        axes = axes.flatten()
        img_array = []
        # select random images
        for container_counter in range(0, (self.rows * self.column)):
            image = cv2.imread(random.choice(image_paths_angry))
            img_array.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        # create grid
        for container_counter in range(0, (self.rows * self.column)):
            axes[2 * container_counter].imshow(img_array[container_counter], cmap='gray')
            axes[2 * container_counter].axis('off')
            pixels = img_array[container_counter].flatten()
            axes[((2 * container_counter) + 1)].hist(pixels, bins=256, color="gray", alpha=0.7)
            axes[((2 * container_counter) + 1)].axis('off')
            axes[((2 * container_counter) + 1)].set_xlabel('Pixel Intensity')
            axes[((2 * container_counter) + 1)].set_ylabel('Frequency')
            axes[((2 * container_counter) + 1)].set_title('Pixel Frequency')

        fig.tight_layout(pad=2.0)
        plt.show()
