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
from skimage import transform

data_path = "../data"
output_path = "../output/thresholding/medium/"

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
def plot_images(slices, folder, name, rows=4, cols=6, start=0, freq=1):

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
    plt.savefig("{}/{}.jpg".format(folder, name))

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

    # Use the threshold to filter possible lesions
    max_value, min_value = 1.0, 0.3
    thresh_img = np.where(image > threshold, max_value, min_value)

    # Erode and then dilate regions to remove small anomalies
    eroded = morphology.erosion(thresh_img, np.ones([8, 8]))
    dilation = morphology.dilation(eroded, np.ones([8, 8]))

    # Return the original image if pixels are too far from cluster centers or all intensity values are less than the threshold
    inertia = kmeans.inertia_
    if inertia > 750 or np.max(dilation) < max_value:
        original = exposure.rescale_intensity(original, out_range=(0.0, 1.0)) * 0.3
        original[0][0] = 1.0
        return original

    # Apply the dilated regions to the original image
    return dilation * original

# Resize the image
def resize(images, size):

    resized_images = []

    # Resize the images
    for i in range(len(images)):
        image = images[i]
        size_x = float(len(image))
        size_y = float(len(image[0]))
        new_size_x = size
        new_size_y = (size_y / size_x) * new_size_x
        new_size = (int(new_size_x), int(new_size_y))
        new_image = transform.resize(image, new_size)
        resized_images.append(new_image)
    
    return resized_images

# Load dicom image data
def load_data(folder, save_path, roi=False):

    # Get dicom image data
    original_slices = get_slices(folder + "/Features")
    original_images = get_pixels(original_slices)
    original_images = resize(original_images, 250)

    # Plot original images
    rows, cols = 4, 6
    freq = max(1, int(len(original_images) / (rows * cols)))
    plot_images(original_images, save_path, "Original", rows=rows, cols=cols, freq=freq)

    if roi:
        # Get dicom image data
        roi_slices = get_slices(folder + "/Labels")
        roi_images = get_pixels(roi_slices)
        roi_images = resize(roi_images, 250)

        # Plot roi images
        rows, cols = 4, 6
        freq = max(1, int(len(roi_images) / (rows * cols)))
        plot_images(roi_images, save_path, "ROI", rows=rows, cols=cols, freq=freq)

    return original_images

# Plot a set of dicom images
def plot(images, save_path, masked=False):    

    # Create the lesion mask
    if masked:
        masked_lesion = []
        for img in images:
            masked_lesion.append(create_lesion_mask(img))
        images = masked_lesion

    # Plot the results
    rows, cols = 4, 6
    freq = max(1, int(len(images) / (rows * cols)))
    plot_images(images, save_path, "Segmented", rows=rows, cols=cols, freq=freq)

# Process a folder, containing features and labels in dicom format
def process_folder(path):
    for folder in glob(path + "/*"):

        # Get the save path
        _, folder_name = os.path.split(folder)
        save_path = output_path + folder_name

        # Create the output directory if it doesn't exist
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # Load image data
        print("Processing data at directory {}".format(folder))
        images = load_data(folder, save_path, roi=True)

        # Plot the images with lesion masks
        plot(images, save_path, masked=True)

# Process data
process_folder(data_path)
