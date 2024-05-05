import os
from skimage import io
from skimage.util import crop

# Define the directory of the images
directory = r'C:\Users\13671\Downloads\tank_save_test_3'

# Crop dimensions as ((top, bottom), (left, right))
crop_dimensions = ((210, 0), (1380, 1530))

# Iterate over every file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".tif"):
        # Construct full file path
        file_path = os.path.join(directory, filename)
        image = io.imread(file_path)
        cropped_image = crop(image, crop_dimensions)
        # Save the cropped image with name beginning as "cropped"
        io.imsave(os.path.join(directory, f"cropped_{filename}"), cropped_image)
        print(f"Cropped and saved: {filename}")

print("===============  Cropping Finished  ===============")