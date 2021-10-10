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

        mx = np.zeros(self.yn.shape, dtype=int)
        tc = np.zeros(self.yn.shape, dtype=bool)
        bc = np.zeros(self.yn.shape, dtype=bool)
        x = []
        y = []
        ind_c = []
        k = 0
        for j, yj in enumerate(self.yn):
            tc[j] = dsgn.inside_top_contact(yj)
            bc[j] = dsgn.inside_bottom_contact(yj)
            if j > 0 and tc[j - 1]:
                ind_c.pop()
            for i, xi in enumerate(self.xn):
                if dsgn.inside(xi, yj):
                    x.append(xi)
                    y.append(yj)
                    mx[j] += 1
                    if i > 0 or not bc[j]:
                        ind_c.append(k)
                    k += 1
        mxc = [mx[j] - int(tc[j]) - int(bc[j]) for j in range(len(self.yn))]
        assert sum(mxc) == len(ind_c)
        self.x = np.array(x)
        self.y = np.array(y)
        self.ind_c = np.array(ind_c)
        
        self.ind_ym = self.ind_c.copy()
        j = 1
        i = 0
        for k, ind in enumerate(self.ind_c):
            if k < mxc[0]:
                continue
            if ind - mxc[j] in self.ind_c:
                self.ind_ym[k] = ind - mxc[j - 1]
            i += 1
            if i == mxc[j]:
                i = 0
                j += 1


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

    msh = Mesh2D(x, y, dsgn)
    plt.figure()
    plt.plot(msh.y, msh.x, 'bo')
    plt.plot(msh.y[msh.ind_c], msh.x[msh.ind_c], 'rx')

    plt.figure()
    plt.plot(msh.y[msh.ind_c], msh.x[msh.ind_c], 'bo')
    plt.plot(msh.y[msh.ind_ym], msh.x[msh.ind_ym], 'rx')
    plt.show()
