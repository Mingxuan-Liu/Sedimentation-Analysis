import numpy as np
from scipy.optimize import minimize
from particle_helper import crop_particle


def abs_error(theta, particle, cropped_image, domain_size, length, width):
    """
    Calculate the absolute error between the experimental image and the particle model.

    Parameters:
    - theta: Rotation angle.
    - particle: The particle model.
    - experimental_image: The experimental image data.
    - config_name: The configuration name for creating the particle.
    - domain_size: Side length of the squared domain for shadow generation.
    - crop_size: Tuple (height, width) defining the size of the cropped area around the particle center.

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
    cropped_particle = crop_particle(shadow_arr, length, width)

    # Calculate and return the absolute error
    return np.sum(np.abs(cropped_image - cropped_particle))


def optimize_rotation_angle(initial_guess, particle, cropped_image, domain_size, length, width):
    """
    Find the rotation angle 'theta' that minimizes the absolute error between the .tif image and the particle model.

    Parameters:
    - initial_theta: Initial guess for the rotation angle.
    - particle: The particle model.
    - experimental_image: The experimental image data.
    - config_name: The configuration name for creating the particle.
    - domain_size: Side length of the squared domain for shadow generation.
    - crop_size: Tuple (height, width) defining the size of the cropped area around the particle center.

    Returns:
    - Optimal rotation angle.
    """
    # Minimize the absolute error
    result = minimize(
        abs_error,  # function to minimize
        initial_guess,  # initial guess for the theta value
        args=(particle, cropped_image, domain_size, length, width),
        # additional arguments for `absolute_error`
        bounds=[(0, 180)]  # bounds for theta
    )

    # Return the optimal rotation angle
    return result.x[0]