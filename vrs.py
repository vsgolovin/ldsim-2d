# -*- coding: utf-8 -*-
"""
Collection of tools for solving van Roosbroeck system.
"""

import numpy as np

def poisson_res(psi, n, p, h, w, eps, eps_0, q, C_dop):
    lhs = -eps[1:-1] * (  1/h[1:  ]          *psi[2:  ]
                       -(1/h[1:  ]+1/h[:-1])*psi[1:-1]
                       + 1/h[ :-1]          *psi[ :-2] )
    rhs = q/eps_0 * (C_dop[1:-1]-n[1:-1]+p[1:-1]) * w 
    r = -lhs+rhs
    return r

def poisson_dF_dpsi(ndot, pdot, h, w, eps, eps_0, q):
    m = len(ndot)-2  # number of inner nodes
    J = np.zeros((3, m))  # Jacobian in tridiagonal form
    J[0, 1:  ] =  eps[1:-2] / h[1:-1]
    J[1,  :  ] = -eps[1:-1] * (1/h[1:]+1/h[:-1]) \
                 + q/eps_0 * (pdot[1:-1] - ndot[1:-1]) * w
    J[2,  :-1] =  eps[2:-1] / h[1:-1]
    return J

def poisson_dF_dphin(ndot, w, eps_0, q):
    J = -q/eps_0 * ndot[1:-1] * w
    return J

def poisson_dF_dphip(pdot, w, eps_0, q):
    J =  q/eps_0 * pdot[1:-1] * w
    return J
