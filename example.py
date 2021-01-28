# -*- coding: utf-8 -*-
"""
An example that demonstrates calculating band diagram of a
p-i-n diode at forward bias.
"""

import numpy as np
from scipy.sparse import linalg as spla
import matplotlib.pyplot as plt
from diode_data import DiodeData
from sample_diode import sd
import vrs
import flux
import units

# setting up the problem
dd = DiodeData(sd)
m = len(dd.x)-2  # number of internal grid points
dd.make_dimensionless()
dd.solve_equilibrium()  # solves Poisson's equation

# use equilibrium solution as initial values
psi = dd.values['psi_eq'].copy()
phi_n = np.zeros_like(psi)
phi_p = np.zeros_like(psi)

# modelling settings
voltages = np.linspace(0, 2.0, 21)
voltages /= units.V
n_iter_max = 5000  # maximum number of iterations at every step
lam = 1e-1  # damping coefficient
# Newton's method iteration for any voltage is stopped 
# when the number of iterations exceeds n_iter_max
# or average change in solution goes below delta_min
# num_iter tracks number of iterations at every step
num_iter = np.zeros_like(voltages)
delta_min = 1e-7
J_values = np.zeros_like(voltages)

# Newton's method
v_prev = 0
for j, v in enumerate(voltages):
    print("V = %.2f"%(v*units.V))
    dv = v-v_prev

    # setting boundary conditions
    psi[0] -= dv/2
    psi[-1] += dv/2
    phi_n[0] = -v/2
    phi_n[-1] = v/2
    phi_p[0] = -v/2
    phi_p[-1] = v/2

    # iteratively finding new solution
    for i in range(n_iter_max):
        b = -vrs.residual(psi, phi_n, phi_p, dd)
        A = vrs.jacobian(psi, phi_n, phi_p, dd)
        A = A.tocsc()
        dx = spla.spsolve(A, b)
        delta = np.mean(np.abs(dx))
        if delta < delta_min:
            break
        psi[1:-1] += dx[:m]*lam
        phi_n[1:-1] += dx[m:2*m]*lam
        phi_p[1:-1] += dx[2*m:]*lam
    num_iter[j] = i
    v_prev = v

    # calculating current density
    jn = flux.mSG_jn(psi_1=psi[:-1], psi_2=psi[1:],
                     phi_n1=phi_n[:-1], phi_n2=phi_n[1:],
                     x1=dd.x[:-1], x2=dd.x[1:],
                     Nc=dd.midp_values['Nc'], Ec=dd.midp_values['Ec'],
                     Vt=dd.Vt, q=dd.q, mu_n=dd.midp_values['mu_n'])
    jp = flux.mSG_jp(psi_1=psi[:-1], psi_2=psi[1:],
                     phi_p1=phi_p[:-1], phi_p2=phi_p[1:],
                     x1=dd.x[:-1], x2=dd.x[1:],
                     Nv=dd.midp_values['Nv'], Ev=dd.midp_values['Ev'],
                     Vt=dd.Vt, q=dd.q, mu_p=dd.midp_values['mu_p'])
    j_total = -(jn+jp)  # strong fluctuations near contacts
    J_values[j] = j_total.mean()

#%% plotting results

dd.original_units()
voltages *= units.V
x = dd.x*1e4  # converting to micrometers
Ec = dd.values['Ec']
Ev = dd.values['Ev']

plt.figure("Band diagram")
plt.plot(x, Ec-psi*units.V, 'k-', lw=1.0)
plt.plot(x, Ev-psi*units.V, 'k-', lw=1.0)
plt.plot(x, -phi_n*units.V, 'b-', lw=0.5, label=r'$\varphi_n$')
plt.plot(x, -phi_p*units.V, 'r-', lw=0.5, label=r'$\varphi_p$')
plt.xlabel(r'$x$, ($\mu$m)')
plt.ylabel(r'$E$ (eV)')

plt.figure('Current-voltage curve')
plt.plot(voltages, J_values*units.j, marker='.')
plt.xlabel('$V$')
plt.ylabel('$j$ (A / cm$^2$)')

plt.figure('Number of iterations')
plt.plot(voltages, num_iter, marker='.')
plt.xlabel('$V$')
plt.ylabel('Iterations')
