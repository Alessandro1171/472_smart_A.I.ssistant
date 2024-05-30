import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import glob
import random
from PIL import Image


class Focused:
    def __init__(self, path, rows=5, column=3):
        self.rows = rows * 2
        self.column = column
        self.path = path
        image_paths_angry = glob.glob('C:/Users/aless/Documents/angry/train/angry/*.jpg')

    def display_grid(self):
        image_paths_angry = glob.glob('C:/Users/aless/Documents/angry/train/angry/*.jpg')

        colors = ["green", "blue", "black", "red"]
        fig = plt.figure(figsize=(10, 7))
        img_array = []

        for container_counter in range(0, (self.rows + self.column)):
            image = cv2.imread(random.choice(image_paths_angry))
            img_array.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        sub_plot_counter = 0
        for container_counter in range(1, ((self.rows + self.column) + 1)):
            sub_plot_counter = sub_plot_counter + 1
            fig.add_subplot(self.rows, self.column, sub_plot_counter)
            plt.imshow(img_array[container_counter], cmap='gray')
            plt.axis('off')
            sub_plot_counter = sub_plot_counter + 1
            fig.add_subplot(self.rows, self.column, sub_plot_counter)
            pixels = img_array[container_counter].flatten()
            plt.hist(pixels, bins=256, color="gray", alpha=0.7)
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.title('Pixel Frequency')

        plt.show()