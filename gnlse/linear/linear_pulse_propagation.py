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
power = 10 # [W]
tFWHM = 0.100 # Input pulse: pulse duration [ps]
t0 = tFWHM / 2 / np.sqrt(np.log(2))  # for dispersive length calculations

# Optical mode params:
    # Dispersion: derivatives of propagation constant at central wavelength
    # n derivatives of betas are in [ps^n/m]
betas = np.array([-10.4e-3])



setup = gnlse.gnlse.GNLSESetup()
# Numerical parameters
###########################################################################
# number of grid time points
setup.resolution = 2**16
# time window [ps]
setup.time_window = 12.5
# number of distance points to save
setup.z_saves = 1000
# relative tolerance for ode solver
setup.rtol = 1e-6
# absoulte tolerance for ode solver
setup.atol = 1e-6


# Physical parameters
###########################################################################
# Central wavelength [nm]
setup.wavelength = wavelength
# Nonlinear coefficient [1/W/m]
setup.nonlinearity = 0.0

###########################################################################
# Dispersive length
LD = t0 ** 2 / np.abs(betas[0])
# Input pulse: peak power [W]
# Fiber length [m]
setup.fiber_length = 5 * LD
# Type of pulse:  gaussian
setup.pulse_model = gnlse.GaussianEnvelope(power, tFWHM)
# Loss coefficient [dB/m]
loss = 0
# Type of dyspersion operator: build from Taylor expansion
setup.dispersion_model = gnlse.DispersionFiberFromTaylor(loss, betas)

# Type of Ramman scattering function: None (default)
# Selftepening: not accounted
setup.self_steepening = False

# Simulation
###########################################################################
solver = gnlse.gnlse.GNLSE(setup)
sol = solver.run()

# Visualization
###########################################################################

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(5, 15), nrows=3)
_, im1 = gnlse.plot_wavelength_vs_distance(sol, WL_range=[700, 850], ax=ax1, cmap="gnuplot")
# ax1.text(-0.0, 0.05, 'a)')
ax1.set_title("a)" + " "*65)
cbar = fig.colorbar(im1, ax=ax1, location='right', anchor=(0, 0.3))
cbar.set_label('Normalized Spectral Power')

_, im2 = gnlse.plot_wavelength_vs_distance(sol, WL_range=[700, 850], phase="deg", ax=ax2, cmap="hsv")
ax2.set_title("b)" + " "*65)
cbar = fig.colorbar(im2, ax=ax2, location='right', anchor=(0, 0.3))
cbar.set_label('Phase [degrees]')

_, im3 = gnlse.plot_delay_vs_distance(sol, time_range=[-2.0, 2.0], ax=ax3, cmap="gnuplot")
ax3.set_title("c)" + " "*65)
cbar = fig.colorbar(im3, ax=ax3, location='right', anchor=(0, 0.3))
cbar.set_label('Normalized Power Density')

plt.savefig("linear_pulse_b3=0.png", dpi=200, bbox_inches="tight")
plt.show()

