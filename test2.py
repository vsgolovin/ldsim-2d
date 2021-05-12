# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sample_slice import sl
from ld_1d import LaserDiode1D

plt.rc('lines', linewidth=0.7)
plt.rc('figure.subplot', left=0.15, right=0.85)

ld = LaserDiode1D(slc=sl, ar_inds=3,
                  L=3000e-4, w=100e-4,
                  R1=0.95, R2=0.05,
                  lam=0.87e-4, ng=3.9,
                  alpha_i=0.5, beta_sp=1e-4)
ld.gen_nonuniform_mesh(y_ext=[0.5, 0.5])
ld.solve_waveguide(remove_layers=[1,1])
ld.make_dimensionless()
ld.solve_equilibrium()

voltages = np.arange(0, 2.5, 0.1)
s_values = np.zeros_like(voltages)
for i, v in enumerate(voltages):
    print(v, end=', ')
    ld.lasing_init(v)
    fluct = 1
    while fluct>1e-7:
        fluct = ld.lasing_step(0.1, 'mSG')
    print(ld.iterations)
    s_values[i] = ld.sol['S']


ld.original_units()
x = ld.xin * 1e4

plt.figure('Band diagram')
plt.plot(x, ld.yin['Ec']-ld.sol['psi'], color='k')
plt.plot(x, ld.yin['Ev']-ld.sol['psi'], color='k')
plt.plot(x, -ld.sol['phi_n'], 'b:')
plt.plot(x, -ld.sol['phi_p'], 'r:')
# plt.savefig('test2.png', dpi=150)

plt.figure('Concentrations')
plt.plot(x, ld.yin['Ec']-ld.sol['psi'], color='k')
plt.plot(x, ld.yin['Ev']-ld.sol['psi'], color='k')
plt.twinx()
plt.plot(x, ld.sol['n'], 'b-')
plt.plot(x, ld.sol['p'], 'r-')
plt.yscale('log')
