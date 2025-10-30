import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

import gnlse


def calc_power_for_soliton_sech(beta2, gamma, t0):
    return -beta2/(gamma*t0**2)


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
power = None # [W]
tFWHM = 0.100 # Input pulse: pulse duration [ps]
t0 = tFWHM / 2 / np.sqrt(np.log(2))  # for dispersive length calculations

# Optical mode params:
    # Dispersion: derivatives of propagation constant at central wavelength
    # n derivatives of betas are in [ps^n/m]
beta2 = -10.371877e-3
beta3 = 0.0 # -0.262858e-3
betas = np.array([beta2, beta3])
gamma = 0.47715253369 # Nonlinear coefficient [1/W/m]

power = calc_power_for_soliton_sech(beta2*1e-24, gamma, 1.0*tFWHM*1e-12)
print("Required power for soliton: ", power, " W.")

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
setup.nonlinearity = gamma

###########################################################################
# Dispersive length
LD = t0 ** 2 / np.abs(betas[0])
# Input pulse: peak power [W]
# Fiber length [m]
setup.fiber_length = 100
# Type of pulse:  gaussian
m0 = 2 * np.log(1 + np.sqrt(2)) # this factor is because of the internal definition the SechEnvelope function
setup.pulse_model = gnlse.SechEnvelope(power, tFWHM*m0)
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
_, im1 = gnlse.plot_wavelength_vs_distance(sol, WL_range=[750, 800], ax=ax1, cmap="gnuplot")
# ax1.text(-0.0, 0.05, 'a)')
ax1.set_title("a)" + " "*65)
cbar = fig.colorbar(im1, ax=ax1, location='right', anchor=(0, 0.3))
cbar.set_label('Normalized Spectral Power')

_, im2 = gnlse.plot_wavelength_vs_distance(sol, WL_range=[700, 850], phase="deg", ax=ax2, cmap="hsv")
ax2.set_title("b)" + " "*65)
cbar = fig.colorbar(im2, ax=ax2, location='right', anchor=(0, 0.3))
cbar.set_label('Phase [degrees]')

_, im3 = gnlse.plot_delay_vs_distance(sol, time_range=[-1.0, 1.0], ax=ax3, cmap="gnuplot")
ax3.set_title("b)" + " "*65)
cbar = fig.colorbar(im3, ax=ax3, location='right', anchor=(0, 0.3))
cbar.set_label('Normalized Power Density')

plt.savefig(f"soliton_nonlinear_pulse_gamma{gamma}_b3={beta3}.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

plt.plot(sol.t,np.abs(sol.At[0,:]) / np.max(np.abs(sol.At[0,:])), label="z = 0 m")
# plt.plot(sol.t,np.real(sol.At[-1,:]))
plt.plot(sol.t,np.abs(sol.At[-1,:]) / np.max(np.abs(sol.At[-1,:])),'-.', label="z = 100 m; Peak-Power = $P_0$")


setup.pulse_model = gnlse.SechEnvelope(2*power, tFWHM*m0)
solver = gnlse.gnlse.GNLSE(setup)
sol = solver.run()
plt.plot(
    sol.t,
    np.abs(sol.At[-1,:]) / np.max(np.abs(sol.At[-1,:])),
    'g--', label="z = 100 m; Peak-Power = $2P_0$")

setup.pulse_model = gnlse.SechEnvelope(power/2, tFWHM*m0)
solver = gnlse.gnlse.GNLSE(setup)
sol = solver.run()
plt.plot(
    sol.t,
    np.abs(sol.At[-1,:]) / np.max(np.abs(sol.At[-1,:])),
    'm:', label="z = 100 m; Peak-Power = $P_0/2$")

plt.xlabel("Time (ps)")
plt.ylabel("Normalized Envelope")
plt.xlim([-0.5,0.5])
plt.legend(loc="lower left",  prop={'size': 11})
plt.savefig("soliton.pdf")
plt.show()