# -*- coding: utf-8 -*-
"""
Current density calculation.
"""

import numpy as np

def bernoulli(x):
    if isinstance(x, np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.true_divide(x, (np.exp(x)-1))
            y[np.where(x==0)] = 1
    elif x==0:
        y = 1
    else:
        y = x/(np.exp(x)-1)
    return y

def bernoulli_dot(x):
    if isinstance(x, np.ndarray):
        enum = np.exp(x)-x-1
        denom = (np.exp(x)-1)**2
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.true_divide(enum, denom)
            y[np.where(x==0)] = 0.5
    elif x==0:
        y = 0.5
    else:
        y = (np.exp(x)-x-1) / (np.exp(x)-1)**2
    return y

def SG_jn(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n):
    pass