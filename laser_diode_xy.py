# -*- coding: utf-8 -*-
"""
2D (vertical-lateral or x-y) laser diode model.
"""

import numpy as np
import design
import constants as const


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
        self.mesh['xb'] = np.linspace(0, self.dsgn.get_thickness(), nx)
        self.mesh['yb'] = np.linspace(0, self.dsgn.get_width(), ny)
        self._update_mesh()

    def _update_mesh(self):
        xb = self.mesh['xb']         # volume boundaries
        xn = (xb[1:] + xb[:-1]) / 2  # nodes
        hx = xn[1:] - xn[:-1]        # spacing between nodes
        wx = xb[1:] - xb[:-1]        # finite volume sizes
        yb = self.mesh['yb']         # same for y axis
        yn = (yb[1:] + yb[:-1]) / 2
        hy = yn[1:] - yn[:-1]
        wy = yb[1:] - yb[:-1]

        # number of x mesh points for every yn
        self.mesh['mx'] = np.zeros(len(yn), dtype=int)
        self.mesh['tc'] = np.zeros(len(yn), dtype=bool)
        self.mesh['bc'] = np.zeros(len(yn), dtype=bool)
        for j, yj in enumerate(yn):
            for xi in xn:
                if self.dsgn.inside(xi, yj):
                    self.mesh['mx'][j] += 1
                self.mesh['tc'][j] = self.dsgn.inside_top_contact(yj)
                self.mesh['bc'][j] = self.dsgn.inside_bottom_contact(yj)

        self.mesh['xn'] = xn
        self.mesh['hx'] = hx
        self.mesh['wx'] = wx
        self.mesh['yn'] = yn
        self.mesh['hy'] = hy
        self.mesh['wy'] = wy


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sample_design import epi

    dsgn = design.Design2D(epi, 20e-4)
    dsgn.add_trenches(2e-4, 4e-4, 2.0e-4)
    dsgn.set_top_contact(10e-4)
    dsgn.set_bottom_contact(18e-4)

    ld = LaserDiode(dsgn, 2000e-4, 0.3, 0.3, 0.87e-4, 3.9, 0.5, 1e-4)
    ld.gen_uniform_mesh(150, 100)
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
