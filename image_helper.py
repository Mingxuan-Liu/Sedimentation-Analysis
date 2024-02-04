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


def check_polar(frames, params):
    """
    This function batch processes all the .tif image frames, and check whether it contains images with sphere number
    other than 2.
    :param frames: .tif image sequence
    :param params: parameter sets for particle tracking using trackpy package
    :return: printed message of whether there is a bad frame in the list
    """
    data = tp.batch(frames, **params)
    # Link the particles in each frame to their positions in previous two frames
    link_data = tp.link(data, 15, memory=3)

    # Group by 'frame' and count the number of particles in each group
    group_sizes = link_data.groupby('frame').size()
    # Find frames where only one particle is identified
    one_particle_frames = group_sizes[group_sizes == 1]
    # Find frames where three particles are identified
    three_particle_frames = group_sizes[group_sizes == 3]
    # Print the frame numbers
    print("Frames below have identified only one particle:")
    print(one_particle_frames.index.tolist())
    print("Frames below have identified three particles:")
    print(three_particle_frames.index.tolist())

    return


def standardize(frame):
    """
    This function standardizes the bright pixels in the provided image, so that it has normal distribution of intensity.
    :param frame: Provided image to be standardized.
    :return: Image with only the bright pixels standardized.
    """
    bright_pixels = frame[frame > 0]
    mean = np.mean(bright_pixels)
    std = np.std(bright_pixels)

    standardized_pixels = (bright_pixels - mean) / std
    frame[frame > 0] = mean + standardized_pixels

    return frame


