import time
import numpy as np
from skimage import io
from optimizer import optimize_rotation_angle
from image_helper import correct_grayscale
from data_handler import normalize_time
from particle_helper import create_particle
from plot_helper import compare_2d, plot_diagnosis, plot_rotcurve
import warnings
warnings.filterwarnings('ignore')
import json
with open('particle_configurations.json') as f:
    configurations = json.load(f)

# Load the .tif image, and then invert it so that the particle appears white
frames = np.invert(io.imread("1Steel&1Cu_CopperUp_27_6fps_in crop.tif"))
corrected_images = correct_grayscale(frames)[10:325]
plot_diagnosis(corrected_images)

config_name = "dimer-st-cu"
p = create_particle(config_name)

# Spread of the Gaussian function to determine the pixel intensity of the particle (the range [1,2] works the best)
spread = 2
# Initial guess for the rotation angle
initial_theta = 0
# List to store the optimal theta values
optimal_thetas = []

# Record the start time
print("============ Optimization Initiated ============")
start_time = time.time()
# Loop through frames
for fr in range(len(corrected_images)):
    if fr == 0:
        search_range = 10
    else:
        search_range = 5
    # Find the optimal rotation angle
    optimal_theta = optimize_rotation_angle(p, corrected_images[fr], spread, initial_theta, search_range=search_range)
    # Use the found optimal rotation angle as the initial guess for the next frame
    initial_theta = optimal_theta
    # Append the optimal theta to the list
    optimal_thetas.append(optimal_theta)
    print(f"Frame:{fr}, Optimal Theta: {optimal_theta}")

# Record the end time
end_time = time.time()
# Compute and print the elapsed time
elapsed_time = end_time - start_time
print(f"============ Optimization Terminated ============ \n" +
      f"Elapsed time: {elapsed_time:.2f} seconds")

scaled_time = normalize_time(len(corrected_images), frame_rate=6,name_light='st')

# Convert the optimal theta values to radians
optimal_thetas_rad = np.radians(optimal_thetas)

plot_rotcurve(scaled_time, abs(optimal_thetas_rad))

compare_2d(corrected_images, p, optimal_thetas, spread)