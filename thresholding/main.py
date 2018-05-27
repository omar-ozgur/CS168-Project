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

# Create a lesion mask based on image processing techniques
def create_lesion_mask(image):

    # Save image details
    original = image.astype(np.float)
    rows, cols = image.shape[0], image.shape[1]
    
    # Use the mean and standard deviation to accentuate differences
    mean = np.mean(image)
    std = np.std(image)
    image = image - mean
    image = image / std

    # Find the average, max, and min
    middle = image[int(cols * 0.25):int(cols * 0.75), int(rows * 0.25):int(rows * 0.75)] 
    mean = np.mean(image)  
    max = np.max(image)
    min = np.min(image)

    # Use the KMeans clustering algorithm to find groups of pixels
    kmeans = KMeans(n_clusters=5).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = centers[3]

    # Use inertia to determine how close pixels are to their cluster centers
    inertia = kmeans.inertia_
    if inertia > 750:
        original = exposure.rescale_intensity(original, out_range=(0.0, 1.0)) * 0.3
        original[0][0] = 1.0
        return original

    # Use the threshold to filter possible lesions
    thresh_img = np.where(image > threshold, 1.0, 0.3)

    # Erode and then dilate regions to remove small anomalies
    eroded = morphology.erosion(thresh_img, np.ones([10, 10]))
    dilation = morphology.dilation(eroded, np.ones([10, 10]))

    # Apply the dilated regions to the original image
    return dilation * original

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
def plot(masked=False):

    # Load the saved image
    images = np.load(images_path)

    # Create the lesion mask
    if masked:
        masked_lesion = []
        for img in images:
            masked_lesion.append(create_lesion_mask(img))
        images = masked_lesion

    # Plot the results
    rows, cols = 4, 6
    freq = max(1, int(len(images) / (rows * cols)))
    plot_images(images, rows=rows, cols=cols, freq=freq)

# Load image data
load_data()

# Plot the images with lesion masks
plot(masked=True)
