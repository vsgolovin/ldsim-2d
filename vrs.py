# -*- coding: utf-8 -*-
"""
Collection of tools for solving van Roosbroeck system.
"""

import numpy as np
from scipy.sparse import spdiags
from scipy.sparse import linalg as spla
import carrier_concentrations as cc
import flux
import recombination as rec

#%% 1. Poisson's equation

def poisson_res(psi, phi_n, phi_p, x, xm, eps, eps_0,
                q, C_dop, Nc, Nv, Ec, Ev, Vt):
    n = cc.n(psi, phi_n, Nc, Ec, Vt)
    p = cc.p(psi, phi_p, Nv, Ev, Vt)
    h = x[1:]-x[:-1]
    lhs = eps[1:-1] * (  1/h[1:  ]          *psi[2:  ]
                       -(1/h[1:  ]+1/h[:-1])*psi[1:-1]
                       + 1/h[ :-1]          *psi[ :-2] )
    rhs = q/eps_0 * (C_dop[1:-1]-n[1:-1]+p[1:-1]) * (xm[1:]-xm[:-1])
    r = lhs+rhs
    return r

def poisson_dF_dpsi(psi, phi_n, phi_p, x, xm, eps, eps_0,
                    q, C_dop, Nc, Nv, Ec, Ev, Vt):
    m = len(x)
    h = x[1:]-x[:-1]  # (m-1)
    ndot = cc.dn_dpsi(psi, phi_n, Nc, Ec, Vt)
    pdot = cc.dp_dpsi(psi, phi_p, Nv, Ev, Vt)

    # Jacobian in tridiagonal form
    J = np.zeros((3, m-2))  # excluding boundary nodes
    J[0, 1:  ] =  eps[1:-2] / h[1:-1]
    J[1,  :  ] = -eps[1:-1] * (1/h[1:]+1/h[:-1]) \
                 + q/eps_0 * (xm[1:]-xm[:-1]) * (pdot[1:-1] - ndot[1:-1])
    J[2,  :-1] =  eps[2:-1] / h[1:-1]
    return J

def poisson_dF_dphin(psi, phi_n, phi_p, x, xm, eps, eps_0,
                     q, C_dop, Nc, Nv, Ec, Ev, Vt):
    ndot = cc.dn_dphin(psi, phi_n, Nc, Ec, Vt)
    J = -q * (xm[1:]-xm[:-1]) * ndot[1:-1]
    return J

def poisson_dF_dphip(psi, phi_n, phi_p, x, xm, eps, eps_0,
                     q, C_dop, Nc, Nv, Ec, Ev, Vt):
    pdot = cc.dp_dphip(psi, phi_p, Nv, Ev, Vt)
    J =  q * (xm[1:]-xm[:-1]) * pdot[1:-1]
    return J

#%% 2. Electron current continuity equation

def jn_res(psi, phi_n, phi_p, x, xm, Nc, Nv, Ec, Ev,
           mid_Ec, mid_Nc, mid_mu_n,
           Vt, q, ni, n0, p0, tau_n, tau_p, B, Cn, Cp):
    "Residual of electron current continuity equation."
    # electon current density at xm
    j = flux.mSG_jn(psi_1=psi[ :-1], psi_2=psi[1:  ],
                    phi_n1=phi_n[:-1], phi_n2=phi_n[1:],
                    x1=x[:-1], x2=x[1:],
                    Nc=mid_Nc, Ec=mid_Ec, Vt=Vt, q=q, mu_n=mid_mu_n)

    # recombination rates
    R_srh = rec.srh_R(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, ni, n0, p0,
                      tau_n, tau_p)
    R_rad = rec.rad_R(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, B)
    R_aug = rec.auger_R(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, Cn, Cp)
    R = R_srh + R_rad + R_aug

    # total residual
    r = j[1:]-j[:-1] - q*(xm[1:]-xm[:-1])*R[1:-1]
    return r

def jn_dF_dpsi(psi, phi_n, phi_p, x, xm, Nc, Nv, Ec, Ev,
               mid_Ec, mid_Nc, mid_mu_n, Vt, q,
               ni, n0, p0, tau_n, tau_p, B, Cn, Cp):
    m = len(x)
    J = np.zeros((3, m-2))

    # main diagonal
    d1 = flux.mSG_djn_dpsi1(psi_1=psi[1:-1], psi_2=psi[2:],
                            phi_n1=phi_n[1:-1], phi_n2=phi_n[2:],
                            x1=x[1:-1], x2=x[2:],
                            Nc=mid_Nc[1:], Ec=mid_Ec[1:],
                            Vt=Vt, q=1, mu_n=mid_mu_n[1:])
    d2 = flux.mSG_djn_dpsi2(psi_1=psi[:-2], psi_2=psi[1:-1],
                            phi_n1=phi_n[:-2], phi_n2=phi_n[1:-1],
                            x1=x[:-2], x2=x[1:-1],
                            Nc=mid_Nc[:-1], Ec=mid_Ec[:-1],
                            Vt=Vt, q=1, mu_n=mid_mu_n[:-1])
    d3_srh = rec.srh_dR_dpsi(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt,
                             ni, n0, p0, tau_n, tau_p)
    d3_rad = rec.rad_dR_dpsi(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt,
                             n0, p0, B)
    d3_aug = rec.auger_dR_dpsi(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt,
                               n0, p0, Cn, Cp)
    J[1, :] = d1 - d2 - q*(xm[1:]-xm[:-1])*(d3_srh+d3_rad+d3_aug)[1:-1]

    # top diagonal
    J[0, 1:] = flux.mSG_djn_dpsi2(psi_1=psi[1:-2], psi_2=psi[2:-1],
                                 phi_n1=phi_n[1:-2], phi_n2=phi_n[2:-1],
                                 x1=x[1:-2], x2=x[2:-1],
                                 Nc=mid_Nc[1:-1], Ec=mid_Ec[1:-1],
                                 Vt=Vt, q=1, mu_n=mid_mu_n[1:-1])

    # bottom diagonal
    J[2, :-1] = -flux.mSG_djn_dpsi1(psi_1=psi[1:-2], psi_2=psi[2:-1],
                                    phi_n1=phi_n[1:-2], phi_n2=phi_n[2:-1],
                                    x1=x[1:-2], x2=x[2:-1],
                                    Nc=mid_Nc[1:-1], Ec=mid_Ec[1:-1],
                                    Vt=Vt, q=1, mu_n=mid_mu_n[1:-1])

    return J

def jn_dF_dphin(psi, phi_n, phi_p, x, xm, Nc, Nv, Ec, Ev,
                mid_Ec, mid_Nc, mid_mu_n, Vt, q,
                ni, n0, p0, tau_n, tau_p, B, Cn, Cp):
    m = len(x)
    J = np.zeros((3, m-2))

    # main diagonal
    d1 = flux.mSG_djn_dphin1(psi_1=psi[1:-1], psi_2=psi[2:],
                             phi_n1=phi_n[1:-1], phi_n2=phi_n[2:],
                             x1=x[1:-1], x2=x[2:],
                             Nc=mid_Nc[1:], Ec=mid_Ec[1:],
                             Vt=Vt, q=1, mu_n=mid_mu_n[1:])
    d2 = flux.mSG_djn_dphin2(psi_1=psi[:-2], psi_2=psi[1:-1],
                             phi_n1=phi_n[:-2], phi_n2=phi_n[1:-1],
                             x1=x[:-2], x2=x[1:-1],
                             Nc=mid_Nc[:-1], Ec=mid_Ec[:-1],
                             Vt=Vt, q=1, mu_n=mid_mu_n[:-1])
    d3_srh = rec.srh_dR_dphin(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt,
                              ni, n0, p0, tau_n, tau_p)
    d3_rad = rec.rad_dR_dphin(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt,
                              n0, p0, B)
    d3_aug = rec.auger_dR_dphin(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt,
                                n0, p0, Cn, Cp)
    J[1, :] = d1 - d2 - q*(xm[1:]-xm[:-1])*(d3_srh+d3_rad+d3_aug)[1:-1]

    # top diagonal
    J[0, 1:] = flux.mSG_djn_dphin2(psi_1=psi[1:-2], psi_2=psi[2:-1],
                                  phi_n1=phi_n[1:-2], phi_n2=phi_n[2:-1],
                                  x1=x[1:-2], x2=x[2:-1],
                                  Nc=mid_Nc[1:-1], Ec=mid_Ec[1:-1],
                                  Vt=Vt, q=1, mu_n=mid_mu_n[1:-1])

    # bottom diagonal
    J[2, :-1] = -flux.mSG_djn_dphin1(psi_1=psi[1:-2], psi_2=psi[2:-1],
                                     phi_n1=phi_n[1:-2], phi_n2=phi_n[2:-1],
                                     x1=x[1:-2], x2=x[2:-1],
                                     Nc=mid_Nc[1:-1], Ec=mid_Ec[1:-1],
                                     Vt=Vt, q=1, mu_n=mid_mu_n[1:-1])

    return J

def jn_dF_dphip(psi, phi_n, phi_p, x, xm, Nc, Nv, Ec, Ev,
                mid_Ec, mid_Nc, mid_mu_n, Vt, q,
                ni, n0, p0, tau_n, tau_p, B, Cn, Cp):
    m = len(x)
    J = np.zeros((1, m-2))

    d_srh = rec.srh_dR_dphip(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt,
                             ni, n0, p0, tau_n, tau_p)
    d_rad = rec.rad_dR_dphip(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt,
                             n0, p0, B)
    d_aug = rec.auger_dR_dphip(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt,
                               n0, p0, Cn, Cp)

    J[0, :] = -q*(xm[1:]-xm[:-1]) * (d_srh+d_rad+d_aug)[1:-1]
    return J

#%% 3. Hole current continuity equation

def jp_res(psi, phi_n, phi_p, x, xm, Nc, Nv, Ec, Ev,
           mid_Ev, mid_Nv, mid_mu_p,
           Vt, q, ni, n0, p0, tau_n, tau_p, B, Cn, Cp):
    "Residual of electron current continuity equation."
    # electon current density at xm
    j = flux.mSG_jp(psi_1=psi[ :-1], psi_2=psi[1:  ],
                    phi_p1=phi_p[:-1], phi_p2=phi_p[1:],
                    x1=x[:-1], x2=x[1:],
                    Nv=mid_Nv, Ev=mid_Ev, Vt=Vt, q=q, mu_p=mid_mu_p)

    # recombination rates
    R_srh = rec.srh_R(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, ni, n0, p0,
                      tau_n, tau_p)
    R_rad = rec.rad_R(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, B)
    R_aug = rec.auger_R(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, Cn, Cp)
    R = R_srh + R_rad + R_aug

    # total residual
    r = j[1:]-j[:-1] + q*(xm[1:]-xm[:-1])*R[1:-1]
    return r

def jp_dF_dpsi(psi, phi_n, phi_p, x, xm, Nc, Nv, Ec, Ev,
               mid_Ev, mid_Nv, mid_mu_p, Vt, q,
               ni, n0, p0, tau_n, tau_p, B, Cn, Cp):
    m = len(x)
    J = np.zeros((3, m-2))
    
    # main diagonal
    d1 = flux.mSG_djp_dpsi1(psi_1=psi[1:-1], psi_2=psi[2:],
                            phi_p1=phi_p[1:-1], phi_p2=phi_p[2:],
                            x1=x[1:-1], x2=x[2:],
                            Nv=mid_Nv[1:], Ev=mid_Ev[1:],
                            Vt=Vt, q=1, mu_p=mid_mu_p[1:])
    d2 = flux.mSG_djp_dpsi2(psi_1=psi[:-2], psi_2=psi[1:-1],
                            phi_p1=phi_p[:-2], phi_p2=phi_p[1:-1],
                            x1=x[:-2], x2=x[1:-1],
                            Nv=mid_Nv[:-1], Ev=mid_Ev[:-1],
                            Vt=Vt, q=1, mu_p=mid_mu_p[:-1])
    d3_srh = rec.srh_dR_dpsi(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt,
                             ni, n0, p0, tau_n, tau_p)
    d3_rad = rec.rad_dR_dpsi(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt,
                             n0, p0, B)
    d3_aug = rec.auger_dR_dpsi(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt,
                               n0, p0, Cn, Cp)
    J[1, :] = d1 - d2 + q*(xm[1:]-xm[:-1])*(d3_srh+d3_rad+d3_aug)[1:-1]

    # top diagonal
    J[0, 1:] = flux.mSG_djp_dpsi2(psi_1=psi[1:-2], psi_2=psi[2:-1],
                                  phi_p1=phi_p[1:-2], phi_p2=phi_p[2:-1],
                                  x1=x[1:-2], x2=x[2:-1],
                                  Nv=mid_Nv[1:-1], Ev=mid_Ev[1:-1],
                                  Vt=Vt, q=1, mu_p=mid_mu_p[1:-1])

    # bottom diagonal
    J[2, :-1] = -flux.mSG_djp_dpsi1(psi_1=psi[1:-2], psi_2=psi[2:-1],
                                    phi_p1=phi_p[1:-2], phi_p2=phi_p[2:-1],
                                    x1=x[1:-2], x2=x[2:-1],
                                    Nv=mid_Nv[1:-1], Ev=mid_Ev[1:-1],
                                    Vt=Vt, q=1, mu_p=mid_mu_p[1:-1])

    return J

def jp_dF_dphin(psi, phi_n, phi_p, x, xm, Nc, Nv, Ec, Ev,
                mid_Ev, mid_Nv, mid_mu_p, Vt, q,
                ni, n0, p0, tau_n, tau_p, B, Cn, Cp):
    m = len(x)
    J = np.zeros((1, m-2))
    
    # main diagonal
    d_srh = rec.srh_dR_dphip(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt,
                             ni, n0, p0, tau_n, tau_p)
    d_rad = rec.rad_dR_dphip(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt,
                             n0, p0, B)
    d_aug = rec.auger_dR_dphip(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt,
                               n0, p0, Cn, Cp)
    J[0, :] = q*(xm[1:]-xm[:-1])*(d_srh+d_rad+d_aug)[1:-1]

    return J

def jp_dF_dphip(psi, phi_n, phi_p, x, xm, Nc, Nv, Ec, Ev,
                mid_Ev, mid_Nv, mid_mu_p, Vt, q,
                ni, n0, p0, tau_n, tau_p, B, Cn, Cp):
    m = len(x)
    J = np.zeros((3, m-2))
    
    # main diagonal
    d1 = flux.mSG_djp_dphip1(psi_1=psi[1:-1], psi_2=psi[2:],
                             phi_p1=phi_p[1:-1], phi_p2=phi_p[2:],
                             x1=x[1:-1], x2=x[2:],
                             Nv=mid_Nv[1:], Ev=mid_Ev[1:],
                             Vt=Vt, q=1, mu_p=mid_mu_p[1:])
    d2 = flux.mSG_djp_dphip2(psi_1=psi[:-2], psi_2=psi[1:-1],
                             phi_p1=phi_p[:-2], phi_p2=phi_p[1:-1],
                             x1=x[:-2], x2=x[1:-1],
                             Nv=mid_Nv[:-1], Ev=mid_Ev[:-1],
                             Vt=Vt, q=1, mu_p=mid_mu_p[:-1])
    d3_srh = rec.srh_dR_dphip(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt,
                              ni, n0, p0, tau_n, tau_p)
    d3_rad = rec.rad_dR_dphip(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt,
                              n0, p0, B)
    d3_aug = rec.auger_dR_dphip(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt,
                                n0, p0, Cn, Cp)
    J[1, :] = d1 - d2 + q*(xm[1:]-xm[:-1])*(d3_srh+d3_rad+d3_aug)[1:-1]

    # top diagonal
    J[0, 1:] = flux.mSG_djp_dphip2(psi_1=psi[1:-2], psi_2=psi[2:-1],
                                   phi_p1=phi_p[1:-2], phi_p2=phi_p[2:-1],
                                   x1=x[1:-2], x2=x[2:-1],
                                   Nv=mid_Nv[1:-1], Ev=mid_Ev[1:-1],
                                   Vt=Vt, q=1, mu_p=mid_mu_p[1:-1])

    # bottom diagonal
    J[2, :-1] = -flux.mSG_djp_dphip1(psi_1=psi[1:-2], psi_2=psi[2:-1],
                                     phi_p1=phi_p[1:-2], phi_p2=phi_p[2:-1],
                                     x1=x[1:-2], x2=x[2:-1],
                                     Nv=mid_Nv[1:-1], Ev=mid_Ev[1:-1],
                                     Vt=Vt, q=1, mu_p=mid_mu_p[1:-1])

    return J

#%% Main functions

def residual(psi, phi_n, phi_p, dd):
    """
    Calculate residual of van Roosbroeck system of equations. First and last
    elements of input arrays (`psi`, `phi_n` and `phi_p`) are considered to be
    fixed, i.e. these values represent boundary conditions.

    Parameters
    ----------
    psi : numpy.ndarray
        Electrostatic potential. Should have size `m`, which is equal to the
        number of `dd` grid nodes.
    phi_n : numpy.ndarray
        Electron quasi-Fermi potential. `phi_n.shape == (m,)`.
    phi_p : TYPE
        Hole quasi-Fermi potential. `phi_p.shape == (m,)`
    dd : diode_data.DiodeData
        Laser diode model parameters.

    Returns
    -------
    r : numpy.ndarray
        Calculated residual. `r.shape == (3*(m-2),)`

    """
    # unpacking the DiodeData object
    # grid
    x = dd.x
    xm = dd.xm

    # constants
    q = dd.q
    Vt = dd.Vt
    eps_0 = dd.eps_0

    # parameters at grid nodes
    Ev = dd.values['Ev']
    Ec = dd.values['Ec']
    C_dop = dd.values['C_dop']
    Nc = dd.values['Nc']
    Nv = dd.values['Nv']
    ni = dd.values['ni']
    n0 = dd.values['n0']
    p0 = dd.values['p0']
    tau_n = dd.values['tau_n']
    tau_p = dd.values['tau_p']
    B = dd.values['B']
    Cn = dd.values['Cn']
    Cp = dd.values['Cp']
    eps = dd.values['eps']

    # parameters at grid midpoints (finite volume boundaries)
    mid_Ev = dd.midp_values['Ev']
    mid_Ec = dd.midp_values['Ec']
    mid_Nc = dd.midp_values['Nc']
    mid_Nv = dd.midp_values['Nv']
    mid_mu_n = dd.midp_values['mu_n']
    mid_mu_p = dd.midp_values['mu_p']

    # calculation
    m = len(x)-2
    r = np.zeros(m*3)
    r[:m] = poisson_res(psi, phi_n, phi_p, x, xm, eps, eps_0, q, C_dop, Nc, Nv,
                        Ec, Ev, Vt)
    r[m:2*m] = jn_res(psi, phi_n, phi_p, x, xm, Nc, Nv, Ec, Ev,
                      mid_Ec, mid_Nc, mid_mu_n,
                      Vt, q, ni, n0, p0, tau_n, tau_p, B, Cn, Cp)
    r[2*m:] = jp_res(psi, phi_n, phi_p, x, xm, Nc, Nv, Ec, Ev,
                     mid_Ev, mid_Nv, mid_mu_p, Vt, q,
                     ni, n0, p0, tau_n, tau_p, B, Cn, Cp)

    return r

def jacobian(psi, phi_n, phi_p, dd):
    """
    Calculate Jacobian of van Roosbroeck system of equations. First and last
    elements of input arrays (`psi`, `phi_n` and `phi_p`) are considered to be
    fixed, i.e. these values represent boundary conditions.

    Parameters
    ----------
    psi : numpy.ndarray
        Electrostatic potential. Should have size `m`, which is equal to the
        number of `dd` grid nodes.
    phi_n : numpy.ndarray
        Electron quasi-Fermi potential. `phi_n.shape == (m,)`.
    phi_p : TYPE
        Hole quasi-Fermi potential. `phi_p.shape == (m,)`
    dd : diode_data.DiodeData
        Laser diode model parameters.

    Returns
    -------
    J : scipy.sparse.dia.dia_matrix
        Calculated Jacobian.

    """
    # unpacking the DiodeData object
    # grid
    x = dd.x
    xm = dd.xm

    # constants
    q = dd.q
    Vt = dd.Vt
    eps_0 = dd.eps_0

    # parameters at grid nodes
    Ev = dd.values['Ev']
    Ec = dd.values['Ec']
    C_dop = dd.values['C_dop']
    Nc = dd.values['Nc']
    Nv = dd.values['Nv']
    ni = dd.values['ni']
    n0 = dd.values['n0']
    p0 = dd.values['p0']
    tau_n = dd.values['tau_n']
    tau_p = dd.values['tau_p']
    B = dd.values['B']
    Cn = dd.values['Cn']
    Cp = dd.values['Cp']
    eps = dd.values['eps']

    # parameters at grid midpoints (finite volume boundaries)
    mid_Ev = dd.midp_values['Ev']
    mid_Ec = dd.midp_values['Ec']
    mid_Nc = dd.midp_values['Nc']
    mid_Nv = dd.midp_values['Nv']
    mid_mu_n = dd.midp_values['mu_n']
    mid_mu_p = dd.midp_values['mu_p']

    # calculating psi, phi_n and phi_p Jacobians for every equation
    j11 = poisson_dF_dpsi(psi, phi_n, phi_p, x, xm, eps, eps_0,
                          q, C_dop, Nc, Nv, Ec, Ev, Vt)
    j12 = poisson_dF_dphin(psi, phi_n, phi_p, x, xm, eps, eps_0,
                           q, C_dop, Nc, Nv, Ec, Ev, Vt)
    j13 = poisson_dF_dphip(psi, phi_n, phi_p, x, xm, eps, eps_0,
                           q, C_dop, Nc, Nv, Ec, Ev, Vt)
    j21 = jn_dF_dpsi(psi, phi_n, phi_p, x, xm, Nc, Nv, Ec, Ev,
                     mid_Ec, mid_Nc, mid_mu_n, Vt,
                     q, ni, n0, p0, tau_n, tau_p, B, Cn, Cp)
    j22 = jn_dF_dphin(psi, phi_n, phi_p, x, xm, Nc, Nv, Ec, Ev,
                      mid_Ec, mid_Nc, mid_mu_n, Vt,
                      q, ni, n0, p0, tau_n, tau_p, B, Cn, Cp)
    j23 = jn_dF_dphip(psi, phi_n, phi_p, x, xm, Nc, Nv, Ec, Ev,
                      mid_Ec, mid_Nc, mid_mu_n, Vt,
                      q, ni, n0, p0, tau_n, tau_p, B, Cn, Cp)
    j31 = jp_dF_dpsi(psi, phi_n, phi_p, x, xm, Nc, Nv, Ec, Ev,
                     mid_Ev, mid_Nv, mid_mu_p, Vt,
                     q, ni, n0, p0, tau_n, tau_p, B, Cn, Cp)
    j32 = jp_dF_dphin(psi, phi_n, phi_p, x, xm, Nc, Nv, Ec, Ev,
                      mid_Ev, mid_Nv, mid_mu_p, Vt,
                      q, ni, n0, p0, tau_n, tau_p, B, Cn, Cp)
    j33 = jp_dF_dphip(psi, phi_n, phi_p, x, xm, Nc, Nv, Ec, Ev,
                      mid_Ev, mid_Nv, mid_mu_p, Vt,
                      q, ni, n0, p0, tau_n, tau_p, B, Cn, Cp)

    # assembling a single matrix of nonzero diagonal elements
    m = len(x)-2
    data = np.zeros((11, 3*m))
    data[0, 2*m:   ] = j13
    data[1,   m:2*m] = j12
    data[1, 2*m:   ] = j23
    data[2,    :m  ] = j11[0]
    data[2,   m:2*m] = j22[0]
    data[2, 2*m:   ] = j33[0]
    data[3,    :m  ] = j11[1]
    data[3,   m:2*m] = j22[1]
    data[3, 2*m:   ] = j33[1]
    data[4,    :m  ] = j11[2]
    data[4,   m:2*m] = j22[2]
    data[4, 2*m:   ] = j33[2]
    data[5,    :m  ] = j21[0]
    data[6,    :m  ] = j21[1]
    data[6,   m:2*m] = j32
    data[7,    :m  ] = j21[2]
    data[8,    :m  ] = j31[0]
    data[9,    :m  ] = j31[1]
    data[10,   :m  ] = j31[2]

    # creating a sparse matrix
    indices = [2*m, m, 1, 0, -1, -m+1, -m, -m-1, -2*m+1, -2*m, -2*m-1]
    J = spdiags(data=data, diags=indices, m=3*m, n=3*m)

    return J

# test run with small direct bias
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from sample_device import sd
    from diode_data import DiodeData

    # initialization
    dd = DiodeData(sd)
    dd.make_dimensionless()
    m = len(dd.x)-2
    dd.solve_equilibrium()

    # initial guess
    psi = dd.values['psi_eq'].copy()
    phi_n = np.zeros_like(psi)
    phi_p = np.zeros_like(psi)
    m = len(psi)-2

    # new boundary conditions
    U1 = -2
    U2 = 2
    psi[0] += U1
    psi[-1] += U2
    phi_n[0] = U1
    phi_n[-1] = U2
    phi_p[0] = U1
    phi_p[-1] = U2

    # Newton's method
    niter = 2000
    lam = 1e-1
    delta = np.zeros(niter)
    for i in range(niter):
        r = residual(psi, phi_n, phi_p, dd)
        J =  jacobian(psi, phi_n, phi_p, dd)
        J = J.tocsr()
        dx = spla.spsolve(J, -r)
        delta[i] = np.mean(np.abs(dx))
        psi[1:-1] += dx[:m]*lam
        phi_n[1:-1] += dx[m:2*m]*lam
        phi_p[1:-1] += dx[2*m:]*lam

    # plotting results
    plt.figure("Convergence")
    plt.plot(delta)
    plt.xlabel('Iteration number')
    plt.yscale('log')

    plt.figure('Band diagram')
    plt.plot(dd.x, dd.values['Ec']-psi, 'k-', lw=1.0)
    plt.plot(dd.x, dd.values['Ev']-psi, 'k-', lw=1.0)
    plt.plot(dd.x, -phi_n, 'b-', lw=0.5)
    plt.plot(dd.x, -phi_p, 'r-', lw=0.5)
    plt.xlabel('$x$')
    plt.ylabel('$E$')