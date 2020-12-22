#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:15:34 2020

@author: vsgolovin
"""

import numpy as np
from scipy.interpolate import interp1d
import units
import constants as const

# DiodeData-specific necessary input parameters
input_params = ['Ev', 'Ec', 'Nd', 'Na', 'Nc', 'Nv', 'mu_n', 'mu_p', 'tau_n',
                'tau_p', 'B', 'Cn', 'Cp', 'eps']
unit_values = {'Ev':units.E, 'Ec':units.E, 'Eg':units.E, 'Nd':units.n,
               'Na':units.n, 'Cdop':units.n, 'Nc':units.n, 'Nv':units.n,
               'mu_n':units.mu, 'mu_p':units.mu, 'tau_n':units.t,
               'tau_p':units.t, 'B':1/(units.n*units.t), 
               'Cn':1/(units.n**2*units.t), 'Cp':1/(units.n**2*units.t),
               'eps':1}

class DiodeData(object):

    def _gen_uni_mesh(self, device, step):
        """
        Generates a uniform mesh. Step may be adjusted if it does not divide
        device thickness.
        """
        width = device.get_thickness()
        assert isinstance(step, (int, float)) and step < width/2
        self.x = np.arange(0, width, step)

    def _gen_nonuni_mesh(self, device, **options):
        """
        Generates a nonuniform mesh.
        """

        def gauss(x, mu, sigma):
            return np.exp( -(x-mu)**2 / (2*sigma**2) )

        # reading optional arguments
        # there is probably a better way to do it
        def _read_option(name, types, default):
            if (name in options
                and isinstance(options[name], types)):
                return options[name]
            return default

        param_grid = _read_option('param_grid', str, 'Eg')
        step_min = _read_option('step_min', float, 1e-7)
        step_max = _read_option('step_max', float, 20e-7)
        sigma = _read_option('sigma', float, 100e-7)
        y0 = _read_option('y0', (int, float, type(None)), 0)
        yn = _read_option('yn', (int, float, type(None)), 0)

        # setting up x and y
        x = self.x.copy()
        if param_grid in device.get_params():
            y = np.array([device.get_value(param_grid, xi) for xi in x])
        elif param_grid=='Eg':
            y = np.array([ ( device.get_value('Ec', xi)
                            -device.get_value('Ev', xi) ) for xi in x])
        else:
            raise Exception('Parameter %s not found'%(param_grid,))

        # adding external values to y
        # same value as inside if y0 or yn is not a number
        if not isinstance(y0, (float, int)):
            y0 = y[0]
        if not isinstance(yn, (float, int)):
            yn = y[-1]
        y_ext = np.concatenate([np.array([y0]), y, np.array([yn])])

        # function for choosing local step size
        f = np.abs(y_ext[2:]-y_ext[:-2])  # change of y at every point
        fg = np.zeros_like(f)  # convolution for smoothing
        for i, xi in enumerate(x):
            g = gauss(x, xi, sigma)
            fg[i] = np.sum(f*g)
        fg_fun = interp1d(x, fg/fg.max())

        # creating new nonuniform grid
        new_grid = list()
        xi = 0
        while xi<=x[-1]:
            new_grid.append(xi)
            xi += step_min + (step_max-step_min)*(1-fg_fun(xi))
        self.x = np.array(new_grid)

    def _calculate_cofficients(self):
        "Calculate rhs coefficients for current continuity and Poisson eqs."
        if self.is_nondimensional:
            self.k_cont = const.q*units.n*units.x / (units.t*units.j)
            self.k_pois = const.q*units.n*units.x**2 / (const.eps_0*units.V)
        else:
            self.k_cont = const.q
            self.k_pois = const.q / const.eps_0

    def __init__(self, device, step=1e-7, uniform=False, **options):
        """
        Class for storing arrays of all the necessary parameters' values.
        Creates a grid at initialization.

        Parameters
        ----------
        device : device.Device
            An object with all the relevant device data.
        step : number
            A step of the created uniform grid (cm). If `uniform=False`, the
            uniform grid is still initially created and used for interpolation
            during nonuniform grid creation.
        uniform : bool
            Whether to use uniform (`True`) or nonuniform (`False`) grid.
        options
            Options passed to a nonuniform grid generator. All the options are
            listed below.
        param_grid : str
            Name of the parameter used for grid generation. Specifically the
            implemented algorithm uses smallest grid spacing in places where
            the chosen parameter experiences large changes.
        step_min : number
            The smallest possible grid step.
        step_max : number
            The largest possible grid step.
        sigma : number
            Standard deviation of the gaussian function, which is used for
            smoothing the parameter change function. Roughly describes the size
            of area that is used to choose the local grid step.
        y0 : number or None
            The 'external' value of parameter 'param_grid' at x<0. Does not any
            physical meaning, added in order to reduce the grid step at the
            boundary. `None` means there is no discontinuity in `param_grid` at
            x=0.
        yn : number or None
            `y0` analogue for the second boundary (x>x_max).
        """
        params = device.get_params()
        assert all([p in params for p in input_params])
        assert device.ready

        # creating 1D mesh
        self._gen_uni_mesh(device, step)
        if not uniform:
            self._gen_nonuni_mesh(device, **options)

        # calculating physical parameters' values at mesh nodes
        self.values = dict()
        n = len(self.x)
        for p in input_params:
            self.values[p] = np.zeros(n)
            for i, xi in enumerate(self.x):
                self.values[p][i] = device.get_value(p, xi)
        # additional parameters
        self.values['Eg'] = self.values['Ec'] - self.values['Ev']
        self.values['Cdop'] = self.values['Nd'] - self.values['Na']

        # tracking measurement units and calculating rhs coefficients
        self.is_nondimensional = False
        self._calculate_cofficients()

    def make_nondimensional(self):
        "Make every parameter dimensionless."
        assert not self.is_nondimensional
        self.x /= units.x
        for key in unit_values:
            self.values[key] /= unit_values[key]
        self.is_nondimensional = True
        self._calculate_cofficients()

    def original_units(self):
        "Return from dimensionless to original units."
        assert self.is_nondimensional
        self.x *= units.x
        for key in unit_values:
            self.values[key] *= unit_values[key]
        self.is_nondimensional = False
        self._calculate_cofficients()

if __name__ == '__main__':
    from sample_device import sd
    import matplotlib.pyplot as plt

    dd = DiodeData(sd)
    dd.make_nondimensional()
    x = dd.x
    Ec = dd.values['Ec']
    Ev = dd.values['Ev']
    mu_p = dd.values['mu_p']
    
    plt.figure()
    plt.plot(x, Ec, 'k-', lw=1.0)
    plt.plot(x, Ev, 'k-', lw=1.0)
    plt.xlabel('$x$')
    plt.ylabel('$E$')
    plt.twinx()
    plt.plot(x, mu_p, 'b:x', ms=4, lw=1.0)
