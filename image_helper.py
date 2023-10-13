import numpy as np
import trackpy as tp


def crop_image(frame, length, width, params):
    """
    Process a .tif image to locate spheres, calculate their center, and crop the image around this center.

    Parameters:
    - frame: one frame of the .tif images to process
    - length: vertical length of the final cropped frame, in pixels
    - width: horizontal width of the final cropped frame, in piexels
    - params: Parameters for tp.locate function (diameter, minmass, etc.)

    Returns:
    - A binary 2D numpy array cropped around the center of the spheres.
    """
    # Locate the spheres
    f = tp.locate(frame, **params)

    # Calculate the center of the spheres
    center_y = np.mean(f['y'])
    center_x = np.mean(f['x'])

    # Define cropping boundaries
    top = int(center_y - length/2)
    bottom = int(center_y + length/2)
    left = int(center_x - width/2)
    right = int(center_x + width/2)

    # Crop the image
    cropped_image = frame[top:bottom, left:right]

    # Convert to binary (0 and 1)
    binary_image = np.where(cropped_image > 0, 1, 0)

    return binary_image
