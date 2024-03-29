import numpy as np
from particle_model import Particle, Sphere
import json
from scipy import ndimage

with open('particle_configurations.json') as f:
    configurations = json.load(f)


def create_particle(name):
    """
    Create and initialize a Particle object based on the provided configuration.

    This function instantiates a Particle object and populates it with Sphere objects.
    The configuration of these Sphere objects, including their center coordinates, radius,
    and material, is retrieved from a pre-defined configuration library using the provided name.

    Parameters
    ----------
    name : str
        The name of the configuration to use when initializing the Particle. This name should
        correspond to a key in the pre-defined configuration library.

    Returns
    -------
    Particle
        The initialized Particle object containing the corresponding spheres based on the given configuration.

    Raises
    ------
    KeyError
        If `name` does not correspond to a key in the configuration library, a KeyError will be raised.
    ValueError
        If any two spheres overlap based on their configuration, a ValueError will be raised.
    """
    # retrieve configuration from the library by name
    configuration = configurations[name]
    p = Particle()  # instantiate a particle object

    # instantiate sphere objects
    for sphere_config in configuration:
        s = Sphere(sphere_config["center"], sphere_config["radius"], sphere_config["material"])
        p.add_sphere(s)

    return p


def find_centroid(arr):
    """
    This function uses the scipy.ndimage.center_of_mass() function to locate the centroid of 2D array (image or shadow)
    :param arr: 2-D array of either the experimental image or the particle shadow
    :return: Indices for the centroid of the 2D array, marked as 1 in the array
    """
    return ndimage.center_of_mass(arr)


def crop_particle(shadow_arr, length, width):
    """
    This function crops the shadow array around the center according to the provided length and width.
    :param shadow_arr: 2D shadow image projected from the 3D particle model
    :param length: vertical length of the output array, in pixels
    :param width: horizontal width of the output array, in pixels
    :return: cropped 2D numpy array
    """
    center = find_centroid(shadow_arr)

    top = int(center[0] - length / 2)
    bottom = int(center[0] + length / 2)
    left = int(center[1] - width / 2)
    right = int(center[1] + width / 2)

    cropped_shadow = shadow_arr[top:bottom, left:right]

    return cropped_shadow



