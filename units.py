# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:10:05 2020

@author: vsgolovin
"""

import constants as const

t = 1e-9
V = const.kb*const.T / 1.0
E = const.kb*const.T
q = const.q
x = q / (const.eps_0*V)
n = 1 / x**3
mu = x**2 / (V*t)
j = q / (t*x**2)
