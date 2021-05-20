# -*- coding: utf-8 -*-

"""
A collection of statistical distribution functions for carrier density
calculation.
"""

import numpy as np
from fdint import fdk, dfdk


def fermi_fdint(nu):
    """
    Fermi-Dirac integral of order 1/2 calculated using `fdint` package.
    """
    F = fdk(k=0.5, phi=nu)
    F *= 2/np.sqrt(np.pi)
    return F


def fermi_dot_fdint(nu):
    """
    Derivative of Fermi-Dirac integral of order 1/2 calculated using `fdint`.
    """
    Fdot = dfdk(k=0.5, phi=nu)
    Fdot *= 2/np.sqrt(np.pi)
    return Fdot

def fermi_approx(nu):
    """
    Fermi-Dirac integral of order 1/2, uses approximate formula from
    "Physics of Photonic Devices" by S.L.Chuang.
    """
    denom = nu + 2.13 + np.power(np.power(np.abs(nu-2.13), 12./5) + 9.6,
                                 5./12)
    C = 3*np.sqrt(np.pi/2) / np.power(denom, 1.5)
    F = 1 / (np.exp(-nu) + C)
    return F

def blakemore(nu):
    """
    Use Blakemore approximation of Fermi-Dirac integral of order 1/2.
    Valid for `nu`<=1.3.
    """
    F = 1 / (np.exp(-nu)+0.27)
    return F

def boltzmann(nu):
    """
    Use Boltzmann approximation of Fermi-Dirac integral of order 1/2.
    Valid for `nu`<=-2.
    """
    F = np.exp(nu)
    return F
