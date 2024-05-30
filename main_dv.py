import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import glob
import random
from PIL import Image

from ClassDistribution import class_distribution, pixel_intensity_pre_class
from SampleImages import SampleImages

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    paths_train = ["C:/Users/aless/Documents/images/angry/train", "C:/Users/aless/Documents/images/focused/train",
                   "C:/Users/aless/Documents/images/happy/train", "C:/Users/aless/Documents/images/neutral/train"]
    paths_test = ["C:/Users/aless/Documents/images/angry/test", "C:/Users/aless/Documents/images/focused/test",
                  "C:/Users/aless/Documents/images/happy/test", "C:/Users/aless/Documents/images/neutral/test"]
    class_distribution(paths_train, paths_test, ["Angry", "Focused", "Happy", "Neutral"])
    pixel_intensity_pre_class(paths_train, paths_test, ["Angry", "Focused", "Happy", "Neutral"])

    sample_images = SampleImages(paths_train[0])
    sample_images.display_grid()
    sample_images = SampleImages(paths_train[1])
    sample_images.display_grid()
    sample_images = SampleImages(paths_train[2])
    sample_images.display_grid()
    sample_images = SampleImages(paths_train[3])
    sample_images.display_grid()

