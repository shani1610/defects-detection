import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import imageio
import os


# Paths to images
image1_path = "data/defective/case2_inspected_image.tif"
image = tiff.imread(image1_path)
is_grayscale = len(image.shape) == 2 or image.shape[2] == 1 # if image.shape has only 2 dimensions → It's grayscale.
print(is_grayscale)