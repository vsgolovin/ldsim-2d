# -*- coding: utf-8 -*-
"""
1D waveguide equation.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs


def solve_wg(x, n, lam, n_modes):
    """
    Solve eigenvalue problem for a 1D waveguide.

    Parameters
    ----------
    x : numpy.ndarray
        x coordinate.
    n : numpy.ndarray
        Refractive index values.
    lam : number
        Wavelength, same units as `x`.
    n_modes : int
        Number of eigenvalues/eigenvectors to be calculated.

    Returns
    -------
    n_eff : numpy.ndarray
        Calculated effective refractive index values.
    modes : numpy.ndarray
        Calculated mode profiles.

    """
    # create matrix A for the eigenvalue problem
    k0 = 2*np.pi / lam
    delta_x = x[1] - x[0]  # uniform mesh
    delta_chi2 = (delta_x * k0)**2
    main_diag = n**2 - 2 / delta_chi2
    off_diag = np.full(x[:-1].shape, 1 / delta_chi2)
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1])

    # solve the eigenproblem
    w, v = eigs(A, k=n_modes, which='SR', sigma=n.max()**2)

    # convert eigenvalues to effective refractive indices
    n_eff = np.sqrt(np.real(w))
    # and eigenvectors to mode profiles
    modes = np.real(v)**2

    # normalize modes
    for i in range(n_modes):
        integral = np.sum(modes[:, i]) * delta_x
        modes[:, i] /= integral

    return n_eff, modes


def solve_wg_2d(x, y, n, lam, n_modes, mx, my):
    """
    Solve eigenvalue problem for a 2D waveguide.
    """
    # create matrix A for the eigenvalue problem
    k0 = 2*np.pi / lam
    dx = x[1] - x[0]
    assert dx > 0
    dy = y[mx] - y[0]
    assert dy > 0
    hx2 = (dx * k0)**2
    hy2 = (dy * k0)**2
    diagonals = np.zeros((5, len(x)))
    diagonals[0, :] = 1 / hy2
    diagonals[1, :] = 1 / hx2
    diagonals[2, :] = n**2 - 2 * (1/hx2 + 1/hy2)
    diagonals[3, :] = 1 / hx2
    diagonals[4, :] = 1 / hy2
    A = diags(diagonals, offsets=[mx, 1, 0, -1, -mx], shape=(mx*my, mx*my))

    # solve the eigenvalue problem
    w, v = eigs(A, k=n_modes, which='SR', sigma=n.max()**2)

    # convert eigenvalues to effective refractive indices
    n_eff = np.sqrt(np.real(w))
    # and eigenvectors to mode profiles
    modes = np.real(v)**2

    # normalize modes
    w = dx * dy
    for i in range(n_modes):
        integral = np.sum(modes[:, i]) * w
        modes[:, i] /= integral

    return n_eff, modes


# example
if __name__=='__main__':
    import matplotlib.pyplot as plt

    # setting up refractive index profile
    x = np.arange(0, 5e-4, 1e-7)
    n = np.full(x.shape, 3.3)
    n[x<2e-4] = 3.1
    n[x>=3e-4] = 3.1
    ix = np.logical_and(x>=2.47e-4, x<2.53e-4)
    n[ix] = 3.6

    # solving eigenvalue problem
    n_eff, modes = solve_wg(x, n, 1.06e-4, 3)

    # plotting results
    plt.figure()
    for i in range(3):
        plt.plot(x, modes[:, i], label=i)
    plt.xlabel('Coordinate')
    plt.ylabel('Mode intensity')
    plt.legend()
    plt.twinx()
    plt.plot(x, n, 'k:', lw=0.8)
    plt.ylabel('Refractive index')
