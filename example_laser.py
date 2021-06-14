# -*- coding: utf-8 -*-
"""
Solving drift-diffusion system coupled with photon density rate equation
at forward bias.
"""

import numpy as np
import matplotlib.pyplot as plt
from sample_design import sd
from laser_diode_xz import LaserDiode

# export settings
export = True
export_folder = 'results'

# set up the problem
ld = LaserDiode(design=sd, ar_inds=10,
                L=3000e-4, w=100e-4,
                R1=0.95, R2=0.05,
                lam=0.87e-4, ng=3.9,
                alpha_i=0.5, beta_sp=1e-4)
ld.gen_nonuniform_mesh(step_min=1e-7, step_max=20e-7, y_ext=[0.5, 0.5])
ld.make_dimensionless()
ld.solve_waveguide(remove_layers=(1, 1))  # ignore contact layers
ld.solve_equilibrium()

# arrays for storing results
voltages = np.arange(0, 2.51, 0.1)
j_values = np.zeros_like(voltages)
I_values = np.zeros_like(voltages)
Isrh_values = np.zeros_like(voltages)
Irad_values = np.zeros_like(voltages)
Iaug_values = np.zeros_like(voltages)
P_values = np.zeros((2, len(voltages)))
print('Voltage Iterations Power')

# solve the transport system for every voltage
# use previous solution as initial guess
for i, v in enumerate(voltages):
    print('  %.2f' % v, end='  ')
    ld.apply_voltage(v)  # apply boundary conditions
    fluct = 1            # fluctuation -- ratio of update vector and
                         # solution L2 norms
    while fluct > 1e-8:  # perform Newton's method iterations
        # choose value of damping parameter `omega`
        # depending on fluctuation
        if fluct > 1e-3:
            omega = 0.05
        else:
            omega = 0.25
        fluct = ld.lasing_step(omega=omega, omega_S=(1.0, omega), discr='mSG')
    # omega -- damping parameter for potentials
    # omega_S -- damping parameters for photon density S
    # `omega_S[0]` is used when increasing S,
    # `omega_S[1]` -- when decreasing S.
    # This is needed so that system does not converge to negative photon
    # density at threshold.

    # save results to corresponding arrays
    j_values[i] = -1 * ld.get_J()
    I_values[i] = -1 * ld.get_I()
    P_values[:, i] = ld.get_P()
    # recombination currents
    Isrh_values[i] = ld.get_Isp('SRH')
    Irad_values[i] = ld.get_Isp('radiative')
    Iaug_values[i] = ld.get_Isp('Auger')

    # save current band diagram and display progress
    if export:
        ld.export_results(folder=export_folder, x_to_um=True)
    print('  %5d    %.1e' % (ld.iterations, P_values[:, i].sum()))

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
plt.ylabel('$E$ (eV)')

plt.figure('Concentrations')
plt.plot(x, ld.yin['Ec']-ld.sol['psi'], color='k')
plt.plot(x, ld.yin['Ev']-ld.sol['psi'], color='k')
plt.xlabel(r'$x$ ($\mu$m)')
plt.ylabel('$E$ (eV)')
plt.twinx()
plt.plot(x, ld.sol['n'], 'b-', label=r'$n$')
plt.plot(x, ld.sol['p'], 'r-', label=r'$p$')
plt.legend()
plt.yscale('log')
plt.ylabel('$n$, $p$ (cm$^-3$)')

plt.figure('J-V curve')
plt.plot(voltages, j_values*1e-3, 'b.-')
plt.xlabel('$V$')
plt.ylabel('$j$ (kA / cm$^2$)')

plt.figure('P-I curve')
plt.plot(I_values, P_values[0, :], 'r.-', label='$P_1$')
plt.plot(I_values, P_values[1, :], 'b.-', label='$P_2$')
plt.legend()
plt.xlabel('$I$ (A)')
plt.ylabel('$P$ (W)')

# export arrays with simulation results
if export:
    with open(export_folder + '/' + 'LIV.csv', 'w') as f:
        f.write(','.join(('V', 'J', 'I', 'P1', 'P2', 'I_srh', 'I_rad', 'I_aug')))
        for i in range(len(voltages)):
            f.write('\n')
            vals = map(str, (voltages[i], j_values[i], I_values[i],
                             P_values[0, i], P_values[1, i],
                             Isrh_values[i], Irad_values[i],
                             Iaug_values[i]))
            f.write(','.join(vals))
