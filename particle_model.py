"""
This program builds a numerical model for a 3-D particle composed of multiple spheres, each of which cannot overlap
and must be connected to at least one another sphere. Basic operations can be performed on the model: rotation, scaling,
and flipping.

@author: Mingxuan Liu
@date: Jul 13, 2023
"""

import numpy as np
from scipy.spatial.transform import Rotation
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
    # The generalized version of the parallel axis theorem can be expressed in the form of
    # coordinate-free notation as J = I + m[(R*R)E_3-R#R]
    # where E_3 is the 3 by 3 identity matrix and # is the outer product.
    I = I_cm + mass * (d**2 * np.eye(3) - np.outer(d_vector, d_vector))
    return I


class Particle:
    def __init__(self):
        self.spheres = []
        self._initial_config = []  # store the initial configuration of the particle, just in case of reset()
        self._center_of_mass = None  # center of mass of the particle
        self._center_of_geometry = None  # geometric center of the particle
        self._offset = None  # the offset 'chi' that measures that distance from com to cog
        self._principal_axes = None  # three principal axes of the particle

    def invalidate_cache(self):
        """
        This function invalidates the cache of computed properties, forcing them to be recalculated the next time
        they're accessed.
        """
        self._center_of_mass = None
        self._center_of_geometry = None
        self._offset = None

    def add_sphere(self, sphere):
        for s in self.spheres:
            if s.is_overlap(sphere):
                raise ValueError("Overlapping spheres are not allowed.")
        self.spheres.append(sphere)
        # Store the initial configuration of the sphere
        self._initial_config.append({
            "center": np.copy(sphere.center),
            "radius": sphere.radius,
            "material": sphere.material
        })
        self.invalidate_cache()

    def reset(self):
        """
        Reset the particle to its original configuration.
        """
        # Clear current sphere objects
        self.spheres.clear()
        # Re-add spheres with initial configurations
        for config in self._initial_config:
            sphere = Sphere(config["center"], config["radius"], config["material"])
            self.spheres.append(sphere)  # Directly append without overlap check for efficiency
        # Invalidate cache to force recalculation of derived properties
        self.invalidate_cache()
        # In each reset, clear the cache for principal axes and recalculate it
        self._principal_axes = None


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

    @property
    def principal_axes(self):
        if self._principal_axes is None:
            # retrieve the inertia tensor property of the particle
            I = self.inertia_tensor()
            # determine the eigenvectors and eigenvalues
            eigenvalues, eigenvectors = np.linalg.eigh(I)
            # normalize the eigenvectors
            eigenvectors /= np.linalg.norm(eigenvectors, axis=0)
            self._principal_axes = eigenvectors  # update the property
        return self._principal_axes

    def inertia_tensor(self):
        """
        Calculate the inertia tensor of the particle.

        :return: Inertia tensor of the particle.
        """
        # Sum up the inertia tensor of each sphere
        I_total = sum([inertia_tensor_sphere(s.mass(), s.radius, s.center - self.center_of_mass) for s in self.spheres])
        return I_total

    def to_numerical_array(self):
        arr = []
        for sphere in self.spheres:
            arr.append([*sphere.center, sphere.radius])
        return np.array(arr)

    def rotate(self, axis, angle):
        """
        Rotate the particle around a given principal axis by a certain angle in degrees.

        Parameters:
        - axis: A string indicating which principal axis to rotate around ('ax1', 'ax2', or 'ax3').
        - angle: The rotation angle in degrees.
        """
        # Convert the angle to radians
        angle_rad = np.radians(angle)

        # Determine the rotation axis based on the principal axes
        if axis == 'ax1':
            rot_axis = self.principal_axes[:, 0]
            other_axes = [1, 2]  # Indices of the other two axes that are perpendicular to the rotating axis
        elif axis == 'ax2':
            rot_axis = self.principal_axes[:, 1]
            other_axes = [0, 2]  # Indices of the other two axes that are perpendicular to the rotating axis
        elif axis == 'ax3':
            rot_axis = self.principal_axes[:, 2]
            other_axes = [0, 1]  # Indices of the other two axes that are perpendicular to the rotating axis
        else:
            raise ValueError("Invalid rotation axis. Choose from 'ax1', 'ax2', or 'ax3'.")

        # Create a rotation object
        rot = Rotation.from_rotvec(angle_rad * rot_axis)

        # Apply the rotation to each sphere in the particle
        for s in self.spheres:
            s.center = rot.apply(s.center - self.center_of_mass) + self.center_of_mass

        # Create a rotation matrix for the principal axes
        # Note: The rotation matrix is transposed because we are transforming coordinate axes
        # (which in this case are the three principal axes)
        rotation_matrix = rot.as_matrix()

        # Rotate only the two principal axes that are perpendicular to the rotation axis
        self._principal_axes[:, other_axes] = np.dot(rotation_matrix, self._principal_axes[:, other_axes])

        # Invalidate the cache to update the properties
        self.invalidate_cache()

    def shadow(self, plane, image_size, center, scale=7):
        """
        Generate a grayscale 2D numpy array representing the shadow of the particle.

        Parameters:
        - plane: A string indicating the projection plane ('xz' or 'yz').
        - image_size: A tuple with the height and width of the experimental image in pixels.
        - center: A tuple with the x and y coordinates of the particle center in mm.
        - scale: The scale factor converting mm to pixels (default is 7).

        Returns:
        - A 2D numpy array where the shadow of the particle is represented with grayscale values.
        """
        # The grid size is now directly taken from the image dimensions
        grid_height, grid_width = image_size

        # Initialize a 2D grid with zeros
        shadow_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

        # Determine the 2D coordinates based on the specified projection plane
        if plane == 'xz':
            indices = (0, 2)  # Use x and z coordinates
        elif plane == 'yz':
            indices = (1, 2)  # Use y and z coordinates
        else:
            raise ValueError("Invalid projection plane. Choose from 'xz' or 'yz'.")

        # Define the center of particle shadow as the centroid of experimental image (floating point)
        center_x, center_y = (center[0], center[1])

        # Iterate over all spheres in the particle configuration
        for s in self.spheres:
            # Read the center coordinate of the sphere from the configuration (floating point)
            sph_x, sph_y = s.center[indices[0]], s.center[indices[1]]
            # Convert to grid coordinates relative to the center of particle
            i = center_x + (sph_x - self.center_of_geometry[indices[0]]) * scale
            j = center_y + (sph_y - self.center_of_geometry[indices[1]]) * scale
            # Mark the grid cell and its neighbors within the radius with grayscale values
            radius_pixels = int(s.radius * scale)
            for di in range(-radius_pixels, radius_pixels + 1):
                for dj in range(-radius_pixels, radius_pixels + 1):
                    distance = np.sqrt(di ** 2 + dj ** 2)  # Calculate the distance for Gaussian function
                    if distance <= radius_pixels:
                        # Apply a Gaussian function to determine the pixel intensity
                        intensity = 255 * np.exp(-(distance ** 2) / (2 * (radius_pixels * 2 / 2) ** 2))
                        shadow_grid[int(i + di), int(j + dj)] = intensity

        return shadow_grid

