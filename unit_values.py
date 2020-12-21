# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:10:05 2020

@author: vsgolovin
"""

import constants as const

x = 1e-4
V = const.kb*const.T / 1.0
E = const.kb*const.T
n = 2e7
mu = 100
j = const.q*mu*n*V/x
R = mu*n*V/x**2
gamma = const.q*n*x**2 / (const.eps_0*V)