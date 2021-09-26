# -*- coding: utf-8 -*-
"""
Mesh operations.
"""

import numpy as np
from scipy.interpolate import interp1d


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
    def __init__(self, x, y, epi, is_dimensionless):
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sample_design import epi

    x = np.linspace(0, epi.get_thickness(), 1000)
    y = epi.calculate('Eg', x)
    xm, ym = generate_nonuniform_mesh(x, y)
    plt.figure()
    plt.plot(x*1e5, y, 'r:', lw=0.5)
    plt.twinx()
    plt.plot(xm*1e5, ym, 'b.-', lw=0.5)
    plt.show()
