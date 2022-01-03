# -*- coding: utf-8 -*-
"""
Mesh operations.
"""

import numpy as np
from scipy.interpolate import interp1d
import design


def generate_nonuniform_mesh(x, y, step_min=1e-7, step_max=20e-7,
                             sigma=100e-7, y_ext=[0, 0]):
    # adding external values to y
    if not isinstance(y_ext[0], (float, int)):
        y_ext[0] = y[0]
    if not isinstance(y_ext[1], (float, int)):
        y_ext[1] = y[-1]

    # absolute value of change in y
    f = np.zeros_like(y)
    f[1:-1] = np.abs(y[2:] - y[:-2])
    f[0] = abs(y[1] - y_ext[0])
    f[-1] = abs(y_ext[1] - y[-2])

    # Gauss function (normal distribution) for smoothing
    g = np.zeros(len(f) * 2)
    i1 = len(f) // 2
    i2 = i1 + len(f)
    g[i1:i2] = np.exp(-(x - x[i1])**2 / (2 * sigma**2))

    # perform convolution for smoothing
    fg = np.zeros_like(f)
    for i in range(len(fg)):
        fg[i] = np.sum(f * g[len(f)-i : len(f)*2-i])
    fg_fun = interp1d(x, fg / fg.max())

    # generate new grid
    new_grid = []
    xi = 0
    while xi <= x[-1]:
        new_grid.append(xi)
        xi += step_min + (step_max - step_min) * (1 - fg_fun(xi))
    xn = np.array(new_grid)
    yn = fg_fun(xn)
    return xn, yn


class Mesh2D:
    def __init__(self, x, y, dsgn):
        self.xb = x
        self.xn = (self.xb[1:] + self.xb[:-1]) / 2
        self.yb = y
        self.yn = (self.yb[1:] + self.yb[:-1]) / 2

        # create masks
        msk = dict()
        for key in ('ins', 'tc', 'bc', 'lb', 'rb', 'bb', 'tb'):
            msk[key] = np.zeros((len(self.xn), len(self.yn)), dtype=bool)
        for j, yj in enumerate(self.yn):
            tc = dsgn.inside_top_contact(yj)
            bc = dsgn.inside_bottom_contact(yj)
            if bc and len(self.xn) > 0:
                msk['bc'][0, j] = True
            for i, xi in enumerate(self.xn):
                if dsgn.inside(xi, yj):
                    msk['ins'][i][j] = True
                else:
                    break
            if tc:
                msk['tc'][i][j] = True
        self.msk = msk


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sample_design import epi

    x = np.linspace(0, epi.get_thickness(), 1000)
    y = epi.calculate('Eg', x)
    xm, _ = generate_nonuniform_mesh(x, y)

    dsgn = design.Design2D(epi, 20e-4)
    dsgn.add_trenches(2e-4, 4e-4, 2e-4)
    dsgn.set_top_contact(10e-4)
    dsgn.set_bottom_contact(18e-4)
    y = np.linspace(0, 20e-4, 100)

    msh = Mesh2D(xm, y, dsgn)
    xn, yn = [], []
    xtc, ytc = [], []
    xbc, ybc = [], []
    for j, yj in enumerate(msh.yn):
        for i, xi in enumerate(msh.xn):
            if msh.msk['ins'][i][j]:
                xn.append(xi)
                yn.append(yj)
            if msh.msk['tc'][i][j]:
                xtc.append(xi)
                ytc.append(yj)
            if msh.msk['bc'][i][j]:
                xbc.append(xi)
                ybc.append(yj)
    plt.figure()
    plt.plot(yn, xn, color='gray', ls='none', marker='o', ms=4)
    plt.plot(ytc, xtc, 'rx', ms=4)
    plt.plot(ybc, xbc, 'bP', ms=4)
    plt.show()
