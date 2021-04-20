# -*- coding: utf-8 -*-
"""
A collection of functions for solving Poisson's equation at equilibrium.
"""

import numpy as np
from scipy.linalg import solve_banded
import carrier_concentrations as cc

def intrinsic_concentration(Nc, Nv, Ec, Ev, Vt):
    "Calculate intrinsic carrier concentration."
    ni = np.sqrt(Nc*Nv)*np.exp( (Ev-Ec) / (2*Vt) )
    return ni

def intrinsic_level(Nc, Nv, Ec, Ev, Vt):
    "Calculate Fermi level location in an intrinsic semiconductor."
    Ei = (Ec+Ev)/2 + Vt/2*np.log(Nv/Nc)
    return Ei

def Ef_lcn_boltzmann(C_dop, ni, Ei, Vt):
    """
    Calculate Fermi level location assuming local charge neutrality and
    Boltzmann statistics.
    """
    xi = np.arcsinh(C_dop/(2*ni))
    Ef = xi*Vt + Ei
    return Ef

def poisson_res(psi, n, p, h, w, eps, eps_0, q, C_dop):
    """
    Calculate residual of Poisson's equation at equilibrium.
    """
    lhs = -eps[1:-1] * (  1/h[1:  ]          *psi[2:  ]
                        -(1/h[1:  ]+1/h[:-1])*psi[1:-1]
                        + 1/h[ :-1]          *psi[ :-2] )
    rhs = q/eps_0 * (C_dop[1:-1] - n[1:-1] + p[1:-1]) * w
    return -lhs+rhs

def poisson_jac(psi, n, ndot, p, pdot, h, w, eps, eps_0, q, C_dop):
    """
    Calculate Jacobian of Poisson's equation at equilibrium.
    """
    m = len(psi)
    J = np.zeros((3, m-2))
    J[0, 1:  ] =  eps[1:-2]/h[1:-1]
    J[1,  :  ] = -eps[1:-1]*(1/h[1:]+1/h[:-1]) \
                 +q/eps_0*(-ndot[1:-1]+pdot[1:-1])*w
    J[2,  :-1] =  eps[2:-1]/h[1:-1]
    return J

def psibi_solve(dd, psi_init, n_iter=3000, lam=1.0, delta_max=1e-6,
                return_delta=False):
    """
    Solve Poisson's equation at equilibrium using Newton's method to find
    built-in potential.

    Parameters
    ----------
    dd : diode_data.DiodeData
        Diode model parameters.
    psi_init : numpy.ndarray
        Initial guess for built-in potential. First and last values are the
        boundary conditions.
    n_iter : int, optional
        Number of Newton's method iterations. The default is 3000.
    lam : float, optional
        Newton's method damping coefficient (0<`lam`<=1). The default is 1.0.
    delta_max : float, optional
        Maximum mean change in solution at last iteration step (relative units).
        The default is 1e-6.
    return_delta : bool, optional
        Whether to return an array of changes in solution at every iterationo.
        The default is False.

    Returns
    -------
    psi : numpy.ndarray
        Calculated built-in potential.
    delta : numpy.ndarray
        Mean relative change in `psi` at every iteration step. Returned only if
        `return_delta` is `True`.

    """
    # unpacking DiodeData object
    x = dd.x    # grid
    xm = dd.xm  # grid midpoints
    q = dd.q    # elementary charge
    eps_0 = dd.eps_0        # vacuum permittivity
    eps = dd.values['eps']  # relative dielectric permittivity
    Nc = dd.values['Nc']    # effective density of states in conduction band
    Nv = dd.values['Nv']    # effective density of states in valence band
    Ec = dd.values['Ec']    # conduction band bottom
    Ev = dd.values['Ev']    # valence band top
    Vt = dd.Vt              # thermal potential
    C_dop = dd.values['C_dop']  # net doping concentration
    psi = psi_init.copy()
    delta = np.zeros(n_iter)

    # Newton's method
    for i in range(n_iter):
        A = poisson_jac(psi, x, xm, eps, eps_0, q, C_dop, Nc, Nv, Ec, Ev, Vt)
        b = poisson_res(psi, x, xm, eps, eps_0, q, C_dop, Nc, Nv, Ec, Ev, Vt)
        dpsi = solve_banded((1, 1), A, -b)
        delta[i] = np.sum(np.abs(dpsi/psi[1:-1])) / (len(x)-2)
        psi[1:-1] += lam*dpsi

    # checking if last step was small
    assert delta[-1]<delta_max

    # returning results
    if return_delta:
        return psi, delta
    else:
        return psi
