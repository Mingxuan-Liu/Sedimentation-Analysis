RADIUS = 0.001  # m, radius of the sphere

# density of the medium in (kg/m^3); in this case the silicone oil, which has specific gravity of 975, compared to the
# density of water 997 kg/m^3.
FLUID_DENSITY = 997 * 0.975

# Densities needed for our sets of particles in (kg/m^3)
DENSITIES = {
    'al': 2790, 
    'st': 7820, 
    'cu': 8920, 
    'pl': 1420, 
    'ZrO2': 5680,
    'wc': 15630
}

# The five terminal velocities needed for our sets of particles in (m/s)
TERMINAL_VELOCITIES = {
    'al': 0.0008167703398558185,
    'st': 0.0030753491246138,
    'cu': 0.0035656345119578895,
    'pl': 0.00020161071060762097,
    'ZrO2': 0.0021144428424304832
}

# The measured terminal velocities
MEASURED_VELOCITIES = {
    'al': 0.0009334337077224641,
    'st': 0.0034170370171462587,
    'cu': 0.0039483145786769315
}

# the characteristic color associated with each type of particle
COLORS = {
    'al': '#C0C0C0',
    'st': '#808080',
    'cu': '#FFA500',
    'pl': '#FFFFFF',
    'ZrO2': '#FFFFFF',
    'wc': '#979A9A'
}