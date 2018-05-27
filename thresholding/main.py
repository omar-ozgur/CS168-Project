# Based on segmentation algorithm described at source https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/

import dicom
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage
from skimage import exposure
from skimage import measure
from skimage import morphology
from sklearn.cluster import KMeans
from skimage.transform import resize

data_path = "./data/FLAIR/1"
output_path = "./output/"
images_name = "images.npy"
images_path = output_path + images_name

# Get individual slices from a dicom image
def get_slices(path):
    slices = [dicom.read_file(file) for file in glob(path + "/*.dcm")]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    return slices

# Get pixels representing each slice of a dicom image
def get_pixels(slices):
    images = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    return np.array(images)

# Plot dicom images as a grid
def plot_images(slices, rows=4, cols=6, start=0, freq=1):

    # Initialize the plot
    figure, grid = plt.subplots(rows, cols, figsize=[cols * 2, rows * 2])
    figure.suptitle("MRI Slices")

    # Create each sublplot
    for i in range(rows * cols):

        # Calulate row and column
        row, col = int(i / cols), int(i % cols)
        if row >= rows or col >= cols:
            break
        index = start + i * freq

        # Plot item
        item = grid[row, col]
        if index < len(slices):
            image = slices[index]
            item.set_title("Slice {}".format(index))
        else:
            image = [[]]
        item.imshow(image, cmap="gray")
        item.axis("off")

    # Display the plot
    plt.show()

# Load dicom image data
def load_data(saved=False, resize=False):
    if saved and images_name not in os.listdir(output_path):
        return np.load(images_path)

    # Get dicom image data
    slices = get_slices(data_path)
    images = get_pixels(slices)

    # Save the processed images for future use
    np.save(images_path, images)

    return images

# Plot a set of dicom images
def plot():

    # Load the saved image
    images = np.load(images_path)

    # Plot the results
    rows, cols = 4, 6
    freq = max(1, int(len(images) / (rows * cols)))
    plot_images(images, rows=rows, cols=cols, freq=freq)

# Load image data
load_data()

# Plot the images with lesion masks
plot()
