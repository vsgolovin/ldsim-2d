# -*- coding: utf-8 -*-
"""
2D (vertical-lateral or x-y) laser diode model.
"""

import numpy as np
from scipy.interpolate import interp1d
import design
import constants as const

params_n = ['Ev', 'Ec', 'Eg', 'Nd', 'Na', 'C_dop', 'Nc', 'Nv', 'mu_n',
            'mu_p', 'tau_n', 'tau_p', 'B', 'Cn', 'Cp', 'eps', 'n_refr',
            'g0', 'N_tr', 'fca_e', 'fca_h']
params_b = ['Ev', 'Ec', 'Nc', 'Nv', 'mu_n', 'mu_p']


class LaserDiode(object):

    def __init__(self, dsgn, L, R1, R2, lam, ng, alpha_i, beta_sp):
        ""
        # check if all the necessary parameters were specified
        # and if there is an active region
        assert isinstance(dsgn, design.Design2D)
        has_active_region = False
        self.ar_inds = list()
        for i, layer in enumerate(dsgn.epi):
            assert True not in [list(yi) == [np.nan] for yi in layer.d.values()]
            if layer.active:
                has_active_region = True
                self.ar_inds.append(i)
        assert has_active_region
        self.dsgn = dsgn

        # constants
        self.Vt = const.kb * const.T
        self.q = const.q
        self.eps_0 = const.eps_0

        # device parameters
        self.L = L
        self.R1 = R1
        self.R2 = R2
        assert all([r > 0 and r < 1 for r in (R1, R2)])
        self.alpha_m = 1/(2*L) * np.log(1/(R1*R2))
        self.lam = lam
        self.photon_energy = const.h * const.c / lam
        self.ng = ng
        self.vg = const.c / ng
        self.n_eff = None
        self.gamma = None
        self.alpha_i = alpha_i
        self.beta_sp = beta_sp

        self.mesh = dict()
        self.vxn = dict()
        self.vxb = dict()
        self.sol = dict()

    def gen_uniform_mesh(self, nx, ny):
        xb = np.linspace(0, self.dsgn.get_thickness(), nx + 1)
        yb = np.linspace(0, self.dsgn.get_width(), ny + 1)
        self.mesh = self._make_mesh(xb, yb)

    def gen_nonuniform_mesh(self, step_min=1e-7, step_max=20e-7, step_uni=5e-8,
                            sigma=100e-7, y_ext=[0., 0.], ny=100):
        def gauss(x, mu, sigma):
            return np.exp(-(x-mu)**2 / (2*sigma**2))

        # uniform y mesh
        yb = np.linspace(0, self.dsgn.get_width(), ny)

        # temporary uniform x mesh
        thickness = self.dsgn.get_thickness()
        nx = int(round(thickness / step_uni))
        x = np.linspace(0, thickness, nx)

        # calculate bandgap
        Eg = np.zeros(len(x) + 2)
        Eg[1:-1] = self.dsgn.epi.calculate('Eg', x)
        # external values for fine grid at boundaries
        for i, j in zip(range(2), [1, -2]):
            if not isinstance(y_ext[i], (float, int)):
                y_ext[i] = Eg[j]
        Eg[0] = y_ext[0]
        Eg[-1] = y_ext[1]

        # function for choosing local step size
        f = np.abs(Eg[2:] - Eg[:-2])  # change of y at every point
        fg = np.zeros_like(f)  # convolution for smoothing
        for i, xi in enumerate(x):
            g = gauss(x, xi, sigma)
            fg[i] = np.sum(f * g)
        fg_fun = interp1d(x, fg / fg.max())

        # generate new mesh
        k = step_max - step_min
        new_mesh = list()
        xi = 0
        while xi <= thickness:
            new_mesh.append(xi)
            xi += step_min + k*(1 - fg_fun(xi))
        xb = np.array(new_mesh)
        self.mesh = self._make_mesh(xb, yb)

    def _make_mesh(self, xb, yb):
        msh = dict()
        msh['xn'] = (xb[1:] + xb[:-1]) / 2          # nodes
        msh['hx'] = msh['xn'][1:] - msh['xn'][:-1]  # spacing between nodes
        msh['wx'] = xb[1:] - xb[:-1]                # finite volume sizes
        msh['yn'] = (yb[1:] + yb[:-1]) / 2
        msh['hy'] = msh['yn'][1:] - msh['yn'][:-1]
        msh['wy'] = yb[1:] - yb[:-1]

        # number of x mesh points for every yn
        my = len(msh['yn'])
        msh['mx'] = np.zeros(my, dtype=int)
        msh['tc'] = np.zeros(my, dtype=bool)
        msh['bc'] = np.zeros(my, dtype=bool)
        for j, yj in enumerate(msh['yn']):
            for xi in msh['xn']:
                if self.dsgn.inside(xi, yj):
                    msh['mx'][j] += 1
                msh['tc'][j] = self.dsgn.inside_top_contact(yj)
                msh['bc'][j] = self.dsgn.inside_bottom_contact(yj)

        return msh


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sample_design import epi

    dsgn = design.Design2D(epi, 20e-4)
    dsgn.add_trenches(2e-4, 4e-4, 2.0e-4)
    dsgn.set_top_contact(10e-4)
    dsgn.set_bottom_contact(18e-4)

    ld = LaserDiode(dsgn, 2000e-4, 0.3, 0.3, 0.87e-4, 3.9, 0.5, 1e-4)
    ld.gen_nonuniform_mesh()
    x = ld.mesh['xn']
    y = ld.mesh['yn']
    mx = ld.mesh['mx']
    x2D = np.zeros(int(mx.sum()))
    y2D = np.zeros_like(x2D)
    i = 0
    for j in range(len(y)):
        x2D[i:i+mx[j]] = x[:mx[j]]
        y2D[i:i+mx[j]] = y[j]
        i += mx[j]

    y_tc = y[ld.mesh['tc']]
    x_tc = np.repeat(x[-1], len(y_tc))
    y_bc = y[ld.mesh['bc']]
    x_bc = np.repeat(x[0], len(y_bc))

    plt.figure()
    plt.plot(y2D*1e4, x2D*1e4, ls='none', marker='.', ms=1, color='b')
    plt.plot(y_tc*1e4, x_tc*1e4, marker='.', ms=2, color='gold', ls='none')
    plt.plot(y_bc*1e4, x_bc*1e4, marker='.', ms=2, color='gold', ls='none')
    plt.show()
