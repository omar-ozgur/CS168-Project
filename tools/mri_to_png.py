import png
import os
import sys
import pydicom
import numpy as np

OUT_DIR = os.getcwd() + "/PNGS"

def convert(mri_path, mri_filename):
    """ Function to convert a DICOM MRI file to PNG

    @param mri_path: The absolute path to the dicom file
    @param mri_filename: The name of the file without the path
    """

    # Read Dicom file
    mri_file = open(mri_path, "rb")
    ds = pydicom.read_file(mri_file)
    mri_file.close()

    # Get dicom data
    shape = ds.pixel_array.shape
    # Convert to float to avoid overflow or underflow losses.
    image_2d = ds.pixel_array.astype(float)
    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
    # Convert to uint8 
    image_2d_scaled = np.uint8(image_2d_scaled)

    # Create PNG file
    png_file = open(os.path.join(OUT_DIR, mri_filename+".png"), "wb")
    # Write to png file
    w = png.Writer(shape[1], shape[0], greyscale=True)
    w.write(png_file, image_2d_scaled)
    png_file.close()


def main():
    # Must provide dicom directory
    if len(sys.argv) < 2:
        print("Please provide an absolute path for the dicom images.")
        sys.exit()
    # Get directory with dicom files from user
    in_dir = sys.argv[1]
    # Create png directory
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    files = []
    # Get all files in dicom directory
    for f in os.listdir(in_dir):
        full_path = os.path.join(in_dir, f)
        if os.path.isfile(full_path):
            files.append((full_path, f[:-4]))
    # Convert all dicom files to png
    for path, name in files:
        convert(path, name)

if __name__ == '__main__':
    main()

