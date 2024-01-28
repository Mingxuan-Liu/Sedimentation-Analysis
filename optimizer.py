import numpy as np
from scipy.optimize import minimize
from particle_helper import crop_particle


def abs_error(theta, particle, cropped_image, domain_size, boundary):
    """
    Calculate the absolute error between the experimental image and the particle model.

    Parameters:
    - theta: Rotation angle.
    - particle: The particle model.
    - cropped_image: The cropped experimental image data.
    - domain_size: Side length of the squared domain for shadow generation.
    - boundary: Tuple (height, width) defining the size of the cropped area around the particle center.

    Returns:
    - The absolute error between the experimental image and the particle model.
    """
    # Reset the particle to its original configuration
    particle.reset()

    # Rotate the particle
    particle.rotate('ax2', theta)

    # Generate the shadow
    shadow_arr = particle.shadow('xz', domain_size)

    # Crop the shadow around the center
    cropped_particle = crop_particle(shadow_arr, boundary[0], boundary[1])

    # Calculate and return the absolute error
    return np.sum(np.abs(cropped_image - cropped_particle))


def optimize_rotation_angle(particle, cropped_image, domain_size, boundary, initial_guess, search_range=20):
    """
    Find the rotation angle 'theta' that minimizes the absolute error between the .tif image and the particle model.

    Parameters:
    - particle: The particle model.
    - cropped_image: The experimental image data.
    - domain_size: Side length of the squared domain for shadow generation.
    - boundary: Tuple (height, width) defining the size of the cropped area around the particle center.
    - initial_guess: Initial guess for the rotation angle.
    - search_range: Range to search around the initial guess (default is 20).
    - step_size: The step size for traversing through the angles (default is 0.5).

    Returns:
    - Optimal rotation angle.
    """
    min_error = float('inf')  # Initialize minimum error as infinity
    optimal_theta = initial_guess  # Initialize optimal theta as the initial guess

    # Define the search bounds and step size
    lower_bound = max(0, initial_guess - search_range)
    upper_bound = min(180, initial_guess + search_range)
    step_size = 0.5
    # Loop through possible theta values within the search bounds using numpy.arange for fractional steps
    for theta in np.arange(lower_bound, upper_bound + step_size, step_size):
        # Calculate the absolute error for the current theta
        error = abs_error(theta, particle, cropped_image, domain_size, boundary)

        # If the error is smaller than the current minimum, update min_error and optimal_theta
        if error < min_error:
            min_error = error
            optimal_theta = theta

    # Return the optimal rotation angle
    return optimal_theta
