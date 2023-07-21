RADIUS = 0.001  # m, radius of the sphere

FLUID_DENSITY = 971  # density of the medium in (kg/m^3); in this case the silicone oil

# Densities needed for our sets of particles in (kg/m^3)
DENSITIES = {
    'al': 2790, 
    'st': 7820, 
    'cu': 8920, 
    'pl': 1420, 
    'ZrO2': 5680
}

# The five terminal velocities needed for our sets of particles in (m/s)
TERMINAL_VELOCITIES = {
    'al': 0.0008167703398558185,
    'st': 0.0030753491246138,
    'cu': 0.0035656345119578895,
    'pl': 0.00020161071060762097,
    'ZrO2': 0.0021144428424304832
}

# the characteristic color associated with each type of particle
COLORS = {
    'al': 'silver',
    'st': 'grey',
    'cu': 'orange',
    'pl': 'white',
    'ZrO2': 'white'
}
