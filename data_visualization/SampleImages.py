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
        fig = plt.figure(figsize=(15, 10))
        img_array = []
        # select random images
        for container_counter in range(0, (self.rows * self.column)):
            image = cv2.imread(random.choice(image_paths_angry))
            img_array.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        # create image grid
        for container_counter in range(0, (self.rows * self.column)):
            fig.add_subplot(self.rows, self.column, (container_counter+1))
            plt.imshow(img_array[container_counter], cmap='gray')
            plt.axis('off')
        fig.tight_layout(pad=2.0)
        plt.show()
        fig = plt.figure(figsize=(15, 10))
        # create histograms
        for container_counter in range(0, (self.rows * self.column)):
            fig.add_subplot(self.rows , self.column, (container_counter+1))
            pixels = img_array[container_counter].flatten()
            plt.hist(pixels, bins=256, color="gray", alpha=0.7)
            plt.axis('off')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.title('Pixel Frequency')
        fig.tight_layout(pad=2.0)
        plt.show()
