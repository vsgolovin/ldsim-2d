# -*- coding: utf-8 -*-
"""
An example that demonstrates calculating band diagram of a
laser diode at forward bias.
"""

import numpy as np
from scipy.sparse import linalg as spla
import matplotlib.pyplot as plt
from laser_data import LaserData
from sample_laser import sl
import vrs_ld as vrs
import flux
import units
import stim_emission as se

# setting up the problem
ld = LaserData(sl, ar_ind=3, lam=0.87e-4, L=2000e-4, w=100e-4, R1=0.3,
                R2=0.3, ng=3.8, alpha_i=0.0, beta_sp=1e-5, step=5e-8)
ld.generate_nonuniform_mesh(sl, step_min=1e-7, step_max=20e-7)
m = len(ld.x)-2
ld.make_dimensionless()
ld.solve_equilibrium()

# use equilibrium solution as initial values
psi = ld.values['psi_eq'].copy()
phi_n = np.zeros_like(psi)
phi_p = np.zeros_like(psi)

# modelling settings
voltages = np.linspace(0.0, 2.5, 51)
voltages /= units.V
niter_1 = 500    # max iterations for transport problem (inner loop)
niter_2 = 200    # both transport and photon generation (outer loop)
lam_1 = 5e-2     # damping coefficient for transport problem
delta_dd = 1e-6  # min change for transport (drift-diffusion) problem solution
delta_se = 1e-9  # same for photon density problem
S_min = 1        # min photon density when outer loop can break
dt = 1e-2        # dimensionless time step

# arrays for storing modelling results
S_values = np.zeros_like(voltages)  # photon density
J_values = np.zeros_like(voltages)  # current density
P_values = np.zeros_like(voltages)  # output power

# iterating over external voltages
v_prev = 0
S = 0
for j, v in enumerate(voltages):
    print('\nV = %.2f'%(v*units.V))
    print('    S          dS')
    dv = v-v_prev

    # setting boundary conditions
    psi[0] -= dv/2
    psi[-1] += dv/2
    phi_n[0] = -v/2
    phi_n[-1] = v/2
    phi_p[0] = -v/2
    phi_p[-1] = v/2

    # iteratively solving the problem
    for _ in range(niter_2):
        for i in range(niter_1):
            b = -vrs.residual(psi, phi_n, phi_p, S, ld)
            A = vrs.jacobian(psi, phi_n, phi_p, S, ld)
            A = A.tocsc()
            dx = spla.spsolve(A, b)
            delta = np.mean(np.abs(dx))
            if delta<delta_dd:
                break
            psi[1:-1] += dx[:m]*lam_1
            phi_n[1:-1] += dx[m:2*m]*lam_1
            phi_p[1:-1] += dx[2*m:]*lam_1

        # calculating change in photon density
        dS = se.delta_S(psi, phi_n, phi_p, S, ld, delta_t=dt)
        if (dS==0) or (S>S_min and dS/S<delta_se):
            break
        print("%.3e  %.3e"% (S, dS))
        S += dS

    v_prev = v
    S_values[j] = S

    # calculating current density
    jn = flux.mSG_jn(psi_1=psi[:-1], psi_2=psi[1:],
                     phi_n1=phi_n[:-1], phi_n2=phi_n[1:],
                     x1=ld.x[:-1], x2=ld.x[1:],
                     Nc=ld.midp_values['Nc'], Ec=ld.midp_values['Ec'],
                     Vt=ld.Vt, q=ld.q, mu_n=ld.midp_values['mu_n'])
    jp = flux.mSG_jp(psi_1=psi[:-1], psi_2=psi[1:],
                     phi_p1=phi_p[:-1], phi_p2=phi_p[1:],
                     x1=ld.x[:-1], x2=ld.x[1:],
                     Nv=ld.midp_values['Nv'], Ev=ld.midp_values['Ev'],
                     Vt=ld.Vt, q=ld.q, mu_p=ld.midp_values['mu_p'])
    j_total = -(jn+jp)
    J_values[j] = j_total.mean()

#%% converting to original units and calculating some additional values
import carrier_concentrations as cc
import constants as const

ld.original_units()
voltages *= units.V
S_values *= units.n*units.x
J_values *= units.j
I_values = J_values*ld.w*ld.L
psi *= units.V
phi_n *= units.V
phi_p *= units.V
n = cc.n(psi, phi_n, ld.values['Nc'], ld.values['Ec'], ld.Vt)
p = cc.p(psi, phi_p, ld.values['Nv'], ld.values['Ev'], ld.Vt)
P_values = S_values*ld.w*ld.L * ld.vg*ld.alpha_m * const.h*const.c/ld.lam
slope = (P_values[1:]-P_values[:-1]) / (I_values[1:]-I_values[:-1])
x = ld.x*1e4  # micrometers

#%% plotting results
plt.close('all')

plt.figure("Band diagram")
plt.plot(x, ld.values['Ec']-psi, 'k-', lw=1.0)
plt.plot(x, ld.values['Ev']-psi, 'k-', lw=1.0)
l1, = plt.plot(x, -phi_n, 'b-', lw=0.5, label=r'$\varphi_n$')
l2, = plt.plot(x, -phi_p, 'r-', lw=0.5, label=r'$\varphi_p$')
plt.legend(handles=[l1, l2])
plt.xlabel(r'$x$ ($\mu$m)')
plt.ylabel(r'$E$ (eV)')

plt.figure('JV curve')
plt.plot(voltages, J_values*units.j, marker='.')
plt.xlabel('$V$')
plt.ylabel('$j$ (A / cm$^2$)')

plt.figure('PI curve')
plt.plot(I_values, P_values, marker='.')
plt.xlabel('$I$ (A)')
plt.ylabel('$P$ (W)')
plt.twinx()
plt.plot((I_values[:-1]+I_values[1:])/2, slope, 'r--', lw=0.5)
plt.ylabel(r'$\eta_d$ (W/A)')

plt.figure('Carrier concentrations')
plt.plot(x, ld.values['Ec']-psi, 'k-', lw=1.0)
plt.plot(x, ld.values['Ev']-psi, 'k-', lw=1.0)
plt.xlabel(r'$x$ ($\mu$m)')
plt.ylabel(r'$E$ (eV)')
plt.twinx()
plt.plot(x, n, 'b-', lw=0.5)
plt.plot(x, p, 'r-', lw=0.5)
plt.yscale('log')
plt.ylim(1e5, 1e19)
plt.ylabel(r'$n$, $p$ (cm$^-3$)')
