"""
This program builds a numerical model for a 3-D particle composed of multiple spheres, each of which cannot overlap
and must be connected to at least one another sphere. Basic operations can be performed on the model: rotation, scaling,
and flipping.

@author: Mingxuan Liu
@date: Jul 13, 2023
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from records import DENSITIES, COLORS  # Import the DENSITIES and COLORS dictionary


class Sphere:
    # Instantiate the sphere by inputting its position, radius, color, and material
    def __init__(self, center, radius, material):
        self.center = np.array(center)
        self.radius = radius
        self.color = COLORS[material]
        self.material = material
        try:
            self.density = DENSITIES[material]  # retrieve the density information from the dictionary
        except KeyError:
            # If not found in the dictionary, then throw an error
            raise ValueError(f"Material '{material}' not found in density records.")

    def is_overlap(self, other):
        # Calculate the distance between one sphere's center to other spheres' center
        distance = np.linalg.norm(self.center - other.center)
        # Return whether the distance is smaller than the sum of two radii
        return distance < (self.radius + other.radius)

    def mass(self):
        # Calculate the mass based on volume and density (mass = density * volume)
        volume = (4 / 3) * np.pi * (self.radius ** 3)  # Volume of sphere
        return (self.density * (10 ** -9)) * volume


class Particle:
    def __init__(self):
        self.spheres = []
        # initialize the center of mass, center of geometry, and the offset 'chi' as none
        self._center_of_mass = None
        self._center_of_geometry = None
        self._offset = None
        # initialize rotation angles
        self.theta = 0  # azimuthal angle
        self.phi = 0  # polar angle

    def invalidate_cache(self):
        """
        This function invalidates the cache of computed properties, forcing them to be recalculated the next time
        they're accessed.
        """
        self._center_of_mass = None
        self._center_of_geometry = None
        self._offset = None

    def add_sphere(self, sphere):
        """
        This function adds spheres objects to the particle object and checks whether it overlaps with the other spheres.

        :param sphere: a sphere object recorded the configuration library
        :return: an updated particle object with newly added spheres
        """
        for s in self.spheres:
            if s.is_overlap(sphere):
                raise ValueError("Overlapping spheres are not allowed.")
        self.spheres.append(sphere)
        # invalidate the cache
        self.invalidate_cache()

    @property
    def center_of_mass(self):
        # if COM does not exist, calculate one
        if self._center_of_mass is None:
            total_mass = sum([s.mass() for s in self.spheres])
            weighted_positions = sum([s.center * s.mass() for s in self.spheres])
            self._center_of_mass = weighted_positions / total_mass
        # if already existed, then read from the cache
        return self._center_of_mass

    @property
    def center_of_geometry(self):
        # if COG does not exist, calculate one
        if self._center_of_geometry is None:
            self._center_of_geometry = sum([s.center for s in self.spheres]) / len(self.spheres)
        # if already existed, then read from the cache
        return self._center_of_geometry

    @property
    def offset(self):
        # if offset does not exist, calculate one
        if self._offset is None:
            distance = np.linalg.norm(self.center_of_mass - self.center_of_geometry)
            average_radius = sum([s.radius for s in self.spheres]) / len(self.spheres)
            self._offset = distance / average_radius  # scale the offset by sphere radius
        # if already existed, then read from the cache
        return self._offset

    def to_numerical_array(self):
        arr = []
        for sphere in self.spheres:
            arr.append([*sphere.center, sphere.radius])
        return np.array(arr)

    def rotate(self, theta, phi):
        """
        This function rotates the particle object by the azimuthal and polar angles theta and phi.

        :param theta: Azimuthal angle for rotation (in degrees)
        :param phi: Polar angle for rotation (in degrees)
        :return: A rotated particle object and updated rotation angles theta and phi
        """
        # define the rotation axis from center of mass to center of geometry
        axis = self.center_of_geometry - self.center_of_mass
        axis = axis / np.linalg.norm(axis)  # normalize the axis

        rotation_azimuth = R.from_rotvec(axis * np.deg2rad(theta))  # create a rotation vector from polar angle

        for s in self.spheres:
            s.center = rotation_azimuth.apply(s.center - self.center_of_mass) + self.center_of_mass

        # update rotation angles
        self.theta += theta
        self.phi += phi

        # invalidate the cache
        self.invalidate_cache()

    def scale(self, factor):
        # scaling is done with respect to the center of mass
        for s in self.spheres:
            s.center = self.center_of_mass + factor * (s.center - self.center_of_mass)
            s.radius = s.radius * factor

        # invalidate the cache
        self.invalidate_cache()

