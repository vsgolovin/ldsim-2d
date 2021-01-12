# -*- coding: utf-8 -*-
"""
A collection of functions for solving the Poisson equation at equilibrium.
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

def Ef_lcn_fermi(C_dop, Nc, Nv, Ec, Ev, Vt, n_iter, lam, adj_max):
    """
    Calculate Fermi level location assuming local charge neutrality and
    Fermi-Dirac statistics.
    """
    # functions for Newton's method
    def f(Ef):
        n = cc.n(Ef, 0, Nc, Ec, Vt)
        p = cc.p(Ef, 0, Nv, Ev, Vt)
        return C_dop-n+p
    def fdot(Ef):
        ndot = cc.dn_dpsi(Ef, 0, Nc, Ec, Vt)
        pdot = cc.dp_dpsi(Ef, 0, Nv, Ev, Vt)
        return -ndot+pdot

    # initial guess
    ni = intrinsic_concentration(Nc, Nv, Ec, Ev, Vt)
    Ei = intrinsic_level(Nc, Nv, Ec, Ev, Vt)
    Ef_i = Ef_lcn_boltzmann(C_dop, ni, Ei, Vt)

    # Newton's method
    for i in range(n_iter):
        delta_Ef = -f(Ef_i)/fdot(Ef_i)
        Ef_i += delta_Ef*lam

    # checking if the last delta_Ef was small
    adj = np.abs(delta_Ef/Ef_i)
    if isinstance(adj, np.ndarray):
        assert adj.mean()<adj_max
    else:
        assert isinstance(adj, (int, float))
        assert adj<adj_max

    return Ef_i

def psibi_initial_guess(dd, n_iter=20, lam=1.0, adj_max=1e-6):
    """
    Calculate built-in potential assuming local charge neutrality and
    Fermi-Dirac statistics. Uses Newton's method with `n_iter` iteration steps
    and `lam` damping coefficient. Asserts the relative change in potential
    during last iteration is below `adj_max`.
    """
    C_dop = dd.values['C_dop']
    Nc = dd.values['Nc']
    Nv = dd.values['Nv']
    Ec = dd.values['Ec']
    Ev = dd.values['Ev']
    Vt = dd.Vt
    psi = Ef_lcn_fermi(C_dop, Nc, Nv, Ec, Ev, Vt, n_iter=n_iter, lam=lam,
                        adj_max=adj_max)
    return psi

def poisson_res(psi, x, xm, eps, eps_0, q, C_dop, Nc, Nv, Ec, Ev, Vt):
    """
    Calculate residual of Poisson's equation at equilibrium.
    """
    n = cc.n(psi, 0, Nc, Ec, Vt)
    p = cc.p(psi, 0, Nv, Ev, Vt)
    h = x[1:]-x[:-1]
    lhs = eps[1:-1] * (  1/h[1:  ]          *psi[2:  ]
                       -(1/h[1:  ]+1/h[:-1])*psi[1:-1]
                       + 1/h[ :-1]          *psi[ :-2] )
    rhs = q/eps_0 * (C_dop[1:-1]-n[1:-1]+p[1:-1]) * (xm[1:]-xm[:-1])
    r = lhs+rhs
    return r

def poisson_jac(psi, x, xm, eps, eps_0, q, C_dop, Nc, Nv, Ec, Ev, Vt):
    """
    Calculate Jacobian of Poisson's equation at equilibrium.
    """
    m = len(psi)
    h = x[1:]-x[:-1]
    ndot = cc.dn_dpsi(psi, 0, Nc, Ec, Vt)
    pdot = cc.dp_dpsi(psi, 0, Nv, Ev, Vt)
    J = np.zeros((3, m-2))
    J[0, 1:  ] =  eps[1:-2]/h[1:-1]
    J[1,  :  ] = -eps[1:-1]*(1/h[1:]+1/h[:-1]) \
                 +q/eps_0*(-ndot[1:-1]+pdot[1:-1])
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
    assert (delta[-1]<delta_max).all()

    # returning results
    if return_delta:
        return psi, delta
    else:
        return psi

# testing convergence
if __name__ == '__main__':
    from diode_data import DiodeData
    from sample_device import sd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    dd = DiodeData(sd)
    dd.make_nondimensional()

    iterations = 5000
    x = dd.x
    psi_init = psibi_initial_guess(dd)
    psi, delta = psibi_solve(dd, psi_init, n_iter=iterations, return_delta=True)

#%% plotting
    plt.close()
    plt.figure('Built-in potential')
    plt.plot(x, psi_init, ls=':', label='initital guess')
    plt.plot(x, psi, ls='-', label='solution')
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel(r'$\psi_{bi}$')

    plt.figure('Band diagram')
    plt.plot(x, dd.values['Ec']-psi_init, 'r:')
    plt.plot(x, dd.values['Ec']-psi, 'r-')
    plt.plot(x, dd.values['Ev']-psi_init, 'b:')
    plt.plot(x, dd.values['Ev']-psi, 'b-')
    l1 = Line2D([], [], ls=':', color='k')
    l2 = Line2D([], [], ls='-', color='k')
    plt.legend(handles=[l1, l2], labels=['initial guess', 'solution'])
    plt.xlabel('$x$')
    plt.ylabel('$E$')

    plt.figure('Convergence')
    plt.semilogy(np.arange(0, iterations), delta)
    plt.xlabel('Iteration number')
    plt.ylabel(r'$\langle \delta \psi_{bi} \rangle$ (rel. u.)')

