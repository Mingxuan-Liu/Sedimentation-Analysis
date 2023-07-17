"""
This program builds a numerical model for a 3-D particle composed of multiple spheres, each of which cannot overlap
and must be connected to at least one another sphere.

@author: Mingxuan Liu
@date: Jul 13, 2023
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from records import DENSITIES  # Import the DENSITIES dictionary


class Sphere:
    # Instantiate the sphere by inputting its position, radius, color, and material
    def __init__(self, center, radius, color, material):
        self.center = np.array(center)
        self.radius = radius
        self.color = color
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
        self._center_of_mass = None
        self._center_of_geometry = None

    def add_sphere(self, sphere):
        for s in self.spheres:
            if s.is_overlap(sphere):
                raise ValueError("Overlapping spheres are not allowed.")
        self.spheres.append(sphere)
        self._center_of_mass = None
        self._center_of_geometry = None

    @property
    def center_of_mass(self):
        if self._center_of_mass is None:
            total_mass = sum([s.mass() for s in self.spheres])
            weighted_positions = sum([s.center * s.mass() for s in self.spheres])
            self._center_of_mass = weighted_positions / total_mass
        return self._center_of_mass

    @property
    def center_of_geometry(self):
        if self._center_of_geometry is None:
            self._center_of_geometry = sum([s.center for s in self.spheres]) / len(self.spheres)
        return self._center_of_geometry

    def to_numerical_array(self):
        arr = []
        for sphere in self.spheres:
            arr.append([*sphere.center, sphere.radius])
        return np.array(arr)

    def rotate(self, angle):
        # define the rotation axis from center of mass to center of geometry
        axis = self.center_of_geometry - self.center_of_mass
        axis = axis / np.linalg.norm(axis)  # normalize the axis

        rotation = R.from_rotvec(axis * np.deg2rad(angle))  # create a rotation vector

        for s in self.spheres:
            s.center = rotation.apply(s.center - self.center_of_mass) + self.center_of_mass

    def scale(self, factor):
        # scaling is done with respect to the center of mass
        for s in self.spheres:
            s.center = self.center_of_mass + factor * (s.center - self.center_of_mass)
            s.radius = s.radius * factor
