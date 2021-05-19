# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sample_design import sd
from ld_1d import LaserDiode1D
import units

plt.rc('lines', linewidth=0.7)
plt.rc('figure.subplot', left=0.15, right=0.85)

ld = LaserDiode1D(design=sd, ar_inds=3,
                  L=3000e-4, w=100e-4,
                  R1=0.95, R2=0.05,
                  lam=0.87e-4, ng=3.9,
                  alpha_i=0.5, beta_sp=1e-4)
ld.gen_nonuniform_mesh(y_ext=[0.5, 0.5])
ld.solve_waveguide(remove_layers=[1,1])
ld.make_dimensionless()
ld.solve_equilibrium()

voltages = np.arange(0, 2.51, 0.1)
j_values = np.zeros_like(voltages)
I_values = np.zeros_like(voltages)
P_values = np.zeros_like(voltages)
for i, v in enumerate(voltages):
    print('%.2f'%v, end=', ')
    ld.lasing_init(v)
    fluct = 1
    while fluct>1e-8:
        fluct = ld.lasing_step(0.1, (1.0, 0.1), 'mSG')
    j_values[i] = -ld.sol['J']
    I_values[i] = ld.sol['I']
    P_values[i] = ld.sol['P']
    print('%d, %.3e' % (ld.iterations, ld.sol['S']))

ld.original_units()
j_values *= units.j
I_values *= units.j * units.x**2
P_values *= units.E / units.t
x = ld.xin * 1e4

plt.figure('Band diagram')
plt.plot(x, ld.yin['Ec']-ld.sol['psi'], color='k')
plt.plot(x, ld.yin['Ev']-ld.sol['psi'], color='k')
plt.plot(x, -ld.sol['phi_n'], 'b:')
plt.plot(x, -ld.sol['phi_p'], 'r:')

plt.figure('Concentrations')
plt.plot(x, ld.yin['Ec']-ld.sol['psi'], color='k')
plt.plot(x, ld.yin['Ev']-ld.sol['psi'], color='k')
plt.twinx()
plt.plot(x, ld.sol['n'], 'b-')
plt.plot(x, ld.sol['p'], 'r-')
plt.yscale('log')

plt.figure('J-V curve')
plt.plot(voltages, j_values)

plt.figure('P-I curve')
plt.plot(I_values, P_values)
