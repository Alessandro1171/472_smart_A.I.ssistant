import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import glob
import random
from PIL import Image


def get_all_images(paths_train: list, paths_test: list):
    """
    get both the training and testing images and combing them into one list for each of the classes
    :param paths_train: list of folder paths for training datasets of each class
    :param paths_test: list of folder paths for testing datasets of each class
    :return: a 2d list containing all the images for each class
    """
    full_paths = []
    for i in range(0, len(paths_train)):
        paths_train_list = glob.glob((paths_train[i] + "/*.png"))

        paths_test_list = glob.glob((paths_test[i] + "/*.png"))
        full_paths.append((paths_train_list + paths_test_list))
    return full_paths


def class_distribution(paths_train: list, paths_test: list, classes: list):
    """
    creates a bar graph showing the number of images in each class
    :param paths_train: list of folder paths for training datasets of each class
    :param paths_test: list of folder paths for testing datasets of each class
    :param classes: list of all classes
    """
    full_paths = get_all_images(paths_train, paths_test)

    list_sizes = [len(section) for section in full_paths]
    fig = plt.figure(figsize=(10, 5))
    plt.bar(classes, list_sizes, color='maroon', width=0.4)
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.title("Class Distribution")
    plt.show()


def pixel_intensity_pre_class(paths_train: list, paths_test: list, classes: list, colors: list):
    """
    plots the aggregated pixel intensity distribution of images for each class in the form of a histogram
    :param colors: number of colors that label each class
    :param paths_train: list of folder paths for training datasets of each class
    :param paths_test: list of folder paths for testing datasets of each class
    :param classes: list of all classes
    """

    global class_counter
    full_paths = get_all_images(paths_train, paths_test)
    fig = plt.figure(figsize=(10, 5))
    sample_size = [len(inner_list) for inner_list in full_paths]
    sides: int = int((len(classes) / 2))
    # construct histogram for each class
    for sample_pool in range(0, min(sample_size)):

        plt.axis('off')
        # construct histogram for specific class
        for class_counter in range(0, len(classes)):
            image = cv2.imread(full_paths[class_counter][sample_pool])
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            pixels = gray_image.flatten()
            plt.hist(pixels, bins=256, color=colors[class_counter], alpha=0.7)

    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    title: str = "Pixel class Intensity Distribution:"
    plt.legend(classes, loc="upper left")
    plt.title(title)
    plt.show()
