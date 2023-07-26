import numpy as np
from particle_model import Particle, Sphere
import json

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


def inertia_tensor_sphere(mass, radius, d_vector):
    """
    Calculate the inertia tensor of a sphere.

    :param mass: Mass of the sphere.
    :param radius: Radius of the sphere.
    :param d_vector: Vector from center of mass of the system to the center of the sphere.
    :return: Inertia tensor of the sphere.
    """
    # Inertia tensor in the center of mass of the sphere
    I_cm = (2/5) * mass * radius**2 * np.eye(3)
    # Distance from the center of mass of the system to the center of the sphere
    d = np.linalg.norm(d_vector)
    # Inertia tensor of the sphere with respect to the system's center of mass
    I = I_cm + mass * d**2 * (np.eye(3) - np.outer(d_vector, d_vector) / d**2)
    return I
