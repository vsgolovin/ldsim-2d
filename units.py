# -*- coding: utf-8 -*-
"""
Unit values for nondimensionalization.
"""

import constants as const

t = 1e-9
E = const.kb*const.T
V = E / 1.0
q = const.q
x = q / (const.eps_0*V)
n = 1 / x**3
mu = x**2 / (V*t)
j = q / (t*x**2)
