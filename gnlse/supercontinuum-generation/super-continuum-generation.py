import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

import gnlse

# Matplotlib Parameters:
rcParams['font.family'] = 'serif'
rcParams['font.size'] = '12'
rcParams['font.style'] = 'normal'
rcParams['font.weight'] = 'medium'
# rcParams['pdf.fonttype'] = '42'
rcParams['grid.linestyle'] = ':'
rcParams['grid.linewidth'] = 1
rcParams['grid.alpha'] = 0.5
rcParams['lines.linewidth'] = 1
# rcParams['axes.autolimit_mode'] = 'round_numbers'
rcParams['axes.xmargin'] = 0
rcParams['axes.ymargin'] = 0.05
rcParams['axes.axisbelow'] = True
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['svg.fonttype'] = 'none'
rcParams["figure.frameon"] = False


# Pulse params:
wavelength = 775 # [nm]
power = 10e3 # [W]
tFWHM = 0.100 # Input pulse: pulse duration [ps]
t0 = tFWHM / 2 / np.sqrt(np.log(2))  # for dispersive length calculations

# Optical mode params:
    # Dispersion: derivatives of propagation constant at central wavelength
    # n derivatives of betas are in [ps^n/m]
beta2 = -10.4e-3
beta3 = 0.0 # -0.263e-3 # -0.263e-3
betas = np.array([beta2, beta3]) # , -9.5205e-8, 2.0737e-10, -5.3943e-13, 1.3486e-15, -2.5495e-18, 3.0524e-21, -1.7140e-24])
# betas = np.array([-11.830e-3, 8.1038e-5, -9.5205e-8, 2.0737e-10, -5.3943e-13, 1.3486e-15, -2.5495e-18, 3.0524e-21, -1.7140e-24])
gamma = 0.477 # Nonlinear coefficient [1/W/m]

raman = False
self_steep = False

setup = gnlse.GNLSESetup()

# Numerical parameters
setup.resolution = 2**14
setup.time_window = 12.5  # ps
setup.z_saves = 500

# Physical parameters
setup.wavelength = wavelength  # nm
setup.fiber_length = 0.1  # m
setup.nonlinearity = 0.11  # 1/W/m
if raman:
    setup.raman_model = gnlse.raman_blowwood
if self_steep:
    setup.self_steepening = True

# The dispersion model is built from a Taylor expansion with coefficients
# given below.
loss = 0

setup.dispersion_model = gnlse.DispersionFiberFromTaylor(loss, betas)

# Input pulse parameters
peak_power = power  # W
duration = tFWHM  # ps

#This example extends the original code with additional simulations for
pulse_models = [
    gnlse.SechEnvelope(peak_power, duration),
    gnlse.GaussianEnvelope(peak_power, duration),
    gnlse.LorentzianEnvelope(peak_power, duration)
]

abc = "abcdef"
count = len(pulse_models)
plt.figure(figsize=(14, 8), facecolor='w', edgecolor='k')
for i, pulse_model in enumerate(pulse_models):
    print('%s...' % pulse_model.name)

    setup.pulse_model = pulse_model
    solver = gnlse.GNLSE(setup)
    solution = solver.run()

    plt.subplot(2, count, i + 1)
    plt.title("("+abc[i]+") Spectral evolution: " + pulse_model.name, fontsize=11)
    gnlse.plot_wavelength_vs_distance(solution, WL_range=[400, 1400])

    plt.subplot(2, count, i + 1 + count)
    plt.title("("+abc[i+3]+") Spatial evolution: " + pulse_model.name, fontsize=11)
    gnlse.plot_delay_vs_distance(solution, time_range=[-1.0, 1.0])

plt.tight_layout()
plt.savefig(f'super_P0={power}_beta3={beta3}_Raman={raman}.png')
plt.show()