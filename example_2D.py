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
# initialize as 1D, because below threshold 1D and 2D models are identical
ld = LaserDiode(design=sd, ar_inds=10,
                L=3000e-4, w=100e-4,
                R1=0.95, R2=0.05,
                lam=0.87e-4, ng=3.9,
                alpha_i=0.5, beta_sp=1e-4)
ld.gen_nonuniform_mesh(step_min=1e-7, step_max=20e-7, y_ext=[0.5, 0.5])
ld.make_dimensionless()
ld.solve_waveguide(remove_layers=(1, 1))  # ignore contact layers
ld.solve_equilibrium()

mz = 10  # number of 1D slices along the longitudinal (z) axis

# arrays for storing results
voltages = np.arange(0, 2.51, 0.1)
mv = len(voltages)
j_values = np.zeros((mz, mv))
Jsrh_values = np.zeros((mz, mv))
Jrad_values = np.zeros((mz, mv))
Jaug_values = np.zeros((mz, mv))
P_values = np.zeros((2, mv))  # store power from both facets separately

# helper function to store results in these arrays
def fill_arrays(i):
    j_values[:, i] = -1 * ld.get_J()
    P_values[:, i] = ld.get_P()
    Jsrh_values[:, i] = ld.get_Jsp('SRH') / mz
    Jrad_values[:, i] = ld.get_Jsp('radiative') / mz
    Jaug_values[:, i] = ld.get_Jsp('Auger') / mz

# solve the transport system for every voltage
# use previous solution as initial guess
print('Voltage Iterations Power')
for i, v in enumerate(voltages):
    print('  %.2f' % v, end='  ')
    ld.apply_voltage(v)  # apply boundary conditions
    fluct = 1            # fluctuation -- ratio of update vector and
                         # solution L2 norms
    while fluct > 5e-9:  # perform Newton's method iterations
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
    fill_arrays(i)

    # save current band diagram and display progress
    if export:
        ld.export_results(folder=export_folder, x_to_um=True)
    print('  %5d    %.1e' % (ld.iterations, P_values[:, i].sum()))
    if P_values[:, i].sum() > 1e-2:
        break

# reached threshold - move to 2D problem
print('Reached threshold. Converting solution to 2D...', end=' ')
ld.to_2D(mz)
fluct = 1.0
while fluct > 1e-9:
    fluct = ld.lasing_step(omega=1.0, omega_S=(1.0, 1.0), discr='mSG')
# update arrays with 2D solution
fill_arrays(i)
print('Complete.')
i += 1

# continue iterating over voltages
while i < mv:
    v = voltages[i]
    print('  %.2f' % v, end='  ')
    ld.apply_voltage(v)
    fluct = 1.0
    while fluct > 1e-11:
        if fluct > 1e-1:
            omega = 0.05
        else:
            omega = 1.0
        fluct = ld.lasing_step(omega=omega, omega_S=(omega, omega), discr='mSG')
    fill_arrays(i)
    if export:
        ld.export_results(folder=export_folder, x_to_um=True)
    print('  %5d    %.1e' % (ld.iterations, P_values[:, i].sum()))
    i += 1

ld.original_units()
x = ld.xin * 1e4
j = j_values.mean(axis=0)
I = j * ld.w * ld.L
I_srh = Jsrh_values.mean(axis=0) * ld.w * ld.L
I_rad = Jrad_values.mean(axis=0) * ld.w * ld.L
I_aug = Jaug_values.mean(axis=0) * ld.w * ld.L

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

plt.figure('J-V curve')
plt.plot(voltages, j*1e-3, 'b.-')
plt.xlabel('$V$')
plt.ylabel('$j$ (kA / cm$^2$)')

plt.figure('P-I curve')
plt.plot(I, P_values[0, :], 'r.-', label='$P_1$')
plt.plot(I, P_values[1, :], 'b.-', label='$P_2$')
plt.legend()
plt.xlabel('$I$ (A)')
plt.ylabel('$P$ (W)')

# export arrays with simulation results
if export:
    with open(export_folder + '/' + 'LIV_2D.csv', 'w') as f:
        f.write(','.join(('V', 'J', 'I', 'P1', 'P2', 'I_srh', 'I_rad', 'I_aug')))
        for i in range(len(voltages)):
            f.write('\n')
            vals = map(str, (voltages[i], j[i], I[i],
                             P_values[0, i], P_values[1, i],
                             I_srh[i], I_rad[i], I_aug[i]))
            f.write(','.join(vals))
