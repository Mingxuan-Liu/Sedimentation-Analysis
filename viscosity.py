from records import DENSITIES, FLUID_DENSITY, RADIUS, MEASURED_VELOCITIES
import numpy as np


# calculates the dynamic viscosity of the fluid based on Eq. 1 in Kavinda's paper
def calc_viscosity(p_density, v_term):
    dyn_viscosity = (2/9) * (p_density - FLUID_DENSITY) * 9.8 * RADIUS ** 2 / v_term
    return dyn_viscosity


# list the name of particles experimented with
p_list = ['al', 'st', 'cu']
etas = np.zeros(len(p_list))  # list of viscosity to be averaged


# enumerate through the list and calculate viscosity for each particle
for i, particle in enumerate(p_list):
    etas[i] = calc_viscosity(DENSITIES[particle], MEASURED_VELOCITIES[particle])
    print(f"Dynamic viscosity measured from {particle}: {etas[i]} Pa*s")

avg_eta = np.average(etas)
print(f"Average dynamic viscosity: {avg_eta} Pa*s")
print(f"Difference with expected viscosity: {((avg_eta - 4.855) / 4.855) * 100:.3f}%")

