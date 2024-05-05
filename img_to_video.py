import os
import numpy as np
from skimage import io
from particle_helper import crop_particle
import imageio

import warnings
warnings.filterwarnings('ignore')

# Define the directory of the images
directory = r'C:\Users\13671\Downloads\tank_save_test_3'

# Initialize an empty variable to store the flat field to be subtracted from other frames
flat_field = []

# Construct an empty array to store the corrected images
frames = []

# Define the output video file name with the directory path
output_file = os.path.join(directory, 'output_video.mp4')

# Define the desired frame size
frame_size = (50, 50)

i = 0
# Iterate over every cropped image in the directory
for filename in os.listdir(directory):
    if filename.startswith("cropped") and filename.endswith(".tif"):
        # Construct full file path
        file_path = os.path.join(directory, filename)
        cropped_frame = np.invert(io.imread(file_path))
        print(f"Processing file: {filename}")

        if i == 0:
            # Define the first image as the flat field reference
            flat_field = np.array(cropped_frame)
        else:
            # Subtract the flat field from the current frame
            corrected_frame = cropped_frame.astype(np.float32) - flat_field.astype(np.float32)

            # Use two thresholds to set fluctuating pixels as 0
            corrected_frame[corrected_frame < 100] = 0
            corrected_frame[corrected_frame > 210] = 0

            # Normalize the corrected frame
            min_val = np.min(corrected_frame)
            max_val = np.max(corrected_frame)
            normalized_frame = 255 * (corrected_frame - min_val) / (max_val - min_val)

            # Crop the normalized frame by a custom size
            cropped_particle = crop_particle(normalized_frame, frame_size[0], frame_size[1])

            # Resize the cropped frame to the desired size
            resized_frame = np.resize(cropped_particle, frame_size)

            frames.append(resized_frame.astype(np.uint8))

        i += 1

# Write the frames to the video using imageio
imageio.mimwrite(output_file, frames, fps=60, macro_block_size=1)