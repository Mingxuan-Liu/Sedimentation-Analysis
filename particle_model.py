"""
This program builds a numerical model for a 3-D particle composed of multiple spheres, each of which cannot overlap
and must be connected to at least one another sphere.

@author: Mingxuan Liu
@date: Jul 13, 2023
"""

import numpy as np
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
        volume = 4 / 3 * np.pi * (self.radius ** 3)  # Volume of sphere
        return self.density * volume


class Particle:
    def __init__(self):
        self.spheres = []

    def add_sphere(self, sphere):
        # Ensure new sphere doesn't overlap existing ones
        for s in self.spheres:
            if s.is_overlap(sphere):
                raise ValueError("Overlapping spheres are not allowed.")
        self.spheres.append(sphere)

    def center_of_mass(self):
        total_mass = sum([s.mass() for s in self.spheres])
        weighted_positions = sum([s.center * s.mass() for s in self.spheres])
        return weighted_positions / total_mass

    def center_of_geometry(self):
        return sum([s.center for s in self.spheres]) / len(self.spheres)

    def to_numerical_array(self):
        arr = []
        for sphere in self.spheres:
            arr.append([*sphere.center, sphere.radius])
        return np.array(arr)

    # Placeholder for future methods
    def rotate(self, axis, angle):
        pass

    def scale(self, factor):
        pass
