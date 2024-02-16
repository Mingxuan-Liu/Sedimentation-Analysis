import numpy as np
from scipy.optimize import minimize
from particle_helper import crop_particle, find_centroid


def abs_error(particle, image):
    """
    Calculate the absolute error between the experimental image and the particle model.

    Parameters:
    - particle: The shadow of the particle model.
    - image: The processed experimental image.

    Returns:
    - The absolute error between the experimental image and the particle model.
    """

    # Calculate and return the absolute error
    return np.sum(np.abs(particle - image))


def optimize_rotation_angle(particle, image, initial_guess, search_range=20):
    """
    Find the rotation angle 'theta' that minimizes the absolute error between the .tif image and the particle model.

    Parameters:
    - particle: The particle model.
    - image: The processed experimental image data.
    - initial_guess: Initial guess for the rotation angle.
    - search_range: Range to search around the initial guess (default is 20).

    Returns:
    - Optimal rotation angle.
    """
    min_error = float('inf')  # Initialize minimum error as infinity
    optimal_theta = initial_guess  # Initialize optimal theta as the initial guess

    # Define the search bounds and step size
    lower_bound = initial_guess - search_range
    upper_bound = initial_guess + search_range
    step_size = 1

    # Define the shape and centroid of the experimental image
    shape = np.shape(image)
    centroid = find_centroid(image)
    # Loop through possible theta values within the search bounds using numpy.arange for fractional steps
    for theta in np.arange(lower_bound, upper_bound + step_size, step_size):
        # Reset the particle to its original configuration
        particle.reset()
        # Rotate the particle
        particle.rotate('ax2', theta)
        # Calculate the shadow array
        shadow_arr = particle.shadow('xz', shape, centroid)
        # Calculate the absolute error for the current theta
        error = abs_error(shadow_arr, image)

        # If the error is smaller than the current minimum, update min_error and optimal_theta
        if error < min_error:
            min_error = error
            optimal_theta = theta

    # Return the optimal rotation angle
    return optimal_theta
