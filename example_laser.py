# -*- coding: utf-8 -*-
"""
Solving drift-diffusion system coupled with photon density rate equation
at forward bias.
"""

import numpy as np
import matplotlib.pyplot as plt
from sample_design import sd
from ld_1d import LaserDiode1D
import units

# export settings
export = True
export_folder = 'results'

# set up the problem
ld = LaserDiode1D(design=sd, ar_inds=3,
                  L=3000e-4, w=100e-4,
                  R1=0.95, R2=0.05,
                  lam=0.87e-4, ng=3.9,
                  alpha_i=0.5, beta_sp=1e-4)
ld.gen_nonuniform_mesh(step_min=1e-7, step_max=20e-7, y_ext=[0.5, 0.5])
ld.make_dimensionless()
ld.solve_waveguide(remove_layers=[1, 1])  # ignore contact layers
ld.solve_equilibrium()

# arrays for storing results
voltages = np.arange(0, 2.51, 0.1)
j_values = np.zeros_like(voltages)
I_values = np.zeros_like(voltages)
Isrh_values = np.zeros_like(voltages)
Irad_values = np.zeros_like(voltages)
Iaug_values = np.zeros_like(voltages)
P_values = np.zeros_like(voltages)
print('Voltage Iterations Power')

# solve the transport system for every voltage
# use previous solution as initial guess
for i, v in enumerate(voltages):
    print('  %.2f' % v, end='  ')
    ld.lasing_init(v)    # apply boundary conditions
    fluct = 1            # fluctuation -- ratio of update vector and
                         # solution L2 norms
    while fluct > 1e-8:  # perform Newton's method iterations
        # choose value of damping parameter `omega`
        # depending on iteration number
        if fluct > 1e-3:
            omega = 0.05
        else:
            omega = 0.25
        fluct = ld.lasing_step(omega=omega, omega_S=(1.0, omega), discr='mSG')
    # omega -- damping parameter for potentials
    # omega_S -- damping parameters for photon density S
    # The first one is used when increasing S, the second one -- when
    # decreasing S.
    # This is needed so that system does not converge to negative photon
    # density at threshold.

    # save results to corresponding arrays
    j_values[i] = -ld.sol['J'] * units.j                 # A/cm2
    I_values[i] = -ld.sol['I'] * (units.j * units.x**2)  # A
    P_values[i] = ld.sol['P'] * (units.E / units.t)      # W
    # recombination currents
    Isrh_values[i] = ld.sol['I_srh'] * (units.j * units.x**2)
    Irad_values[i] = ld.sol['I_rad'] * (units.j * units.x**2)
    Iaug_values[i] = ld.sol['I_aug'] * (units.j * units.x**2)

    # save current band diagram and display progress
    ld.export_results(folder='results', x_to_um=True)
    print('  %5d    %.1e' % (ld.iterations, P_values[i]))

ld.original_units()
x = ld.xin * 1e4

# plot results
# change default Matplotlib settings
plt.rc('lines', linewidth=0.7)
plt.rc('figure.subplot', left=0.15, right=0.85)

plt.figure('Waveguide mode')
plt.plot(x, ld.yin['wg_mode'], 'b-')
plt.ylabel('Mode intensity')
plt.xlabel(r'$x$ ($\mu$m)')
plt.twinx()
plt.plot(x, ld.yin['n_refr'], 'k-', lw=0.5)
plt.ylabel('Refractive index')

plt.figure('Band diagram')
plt.plot(x, ld.yin['Ec']-ld.sol['psi'], color='k')
plt.plot(x, ld.yin['Ev']-ld.sol['psi'], color='k')
plt.plot(x, -ld.sol['phi_n'], 'b:', label=r'$\varphi_n$')
plt.plot(x, -ld.sol['phi_p'], 'r:', label=r'$\varphi_p$')
plt.legend()
plt.xlabel(r'$x$ ($\mu$m)')
plt.ylabel(r'$E$ (eV)')

plt.figure('Concentrations')
plt.plot(x, ld.yin['Ec']-ld.sol['psi'], color='k')
plt.plot(x, ld.yin['Ev']-ld.sol['psi'], color='k')
plt.xlabel(r'$x$ ($\mu$m)')
plt.ylabel(r'$E$ (eV)')
plt.twinx()
plt.plot(x, ld.sol['n'], 'b-', label=r'$n$')
plt.plot(x, ld.sol['p'], 'r-', label=r'$p$')
plt.legend()
plt.yscale('log')
plt.ylabel(r'$n$, $p$ (cm$^-3$)')

plt.figure('J-V curve')
plt.plot(voltages, j_values*1e-3, 'b.-')
plt.xlabel('$V$')
plt.ylabel('$j$ (kA / cm$^2$)')

plt.figure('P-I curve')
plt.plot(I_values, P_values, 'b.-')
plt.xlabel('$I$ (A)')
plt.ylabel('$P$ (W)')

# export arrays with simulation results
with open(export_folder + '/' + 'LIV.csv', 'w') as f:
    f.write(','.join(('V', 'J', 'I', 'P', 'I_srh', 'I_rad', 'I_aug')))
    for i in range(len(voltages)):
        f.write('\n')
        vals = map(str, (voltages[i], j_values[i], I_values[i],
                         P_values[i], Isrh_values[i], Irad_values[i],
                         Iaug_values[i]))
        f.write(','.join(vals))
