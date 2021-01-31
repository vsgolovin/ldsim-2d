# -*- coding: utf-8 -*-
"""
Class for storing all the parameters needed to simulate 1D semiconductor diode
operation.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import solve_banded
import units
import constants as const
import equilibrium as eq
import carrier_concentrations as cc

# DiodeData-specific necessary input parameters
input_params = ['Ev', 'Ec', 'Nd', 'Na', 'Nc', 'Nv', 'mu_n', 'mu_p', 'tau_n',
                'tau_p', 'B', 'Cn', 'Cp', 'eps']
midp_params = ['Ev', 'Ec', 'Nc', 'Nv', 'mu_n', 'mu_p']
unit_values = {'Ev':units.E, 'Ec':units.E, 'Eg':units.E, 'Nd':units.n,
               'Na':units.n, 'C_dop':units.n, 'Nc':units.n, 'Nv':units.n,
               'mu_n':units.mu, 'mu_p':units.mu, 'tau_n':units.t,
               'tau_p':units.t, 'B':1/(units.n*units.t), 
               'Cn':1/(units.n**2*units.t), 'Cp':1/(units.n**2*units.t),
               'eps':1, 'ni':units.n, 'n0':units.n, 'p0':units.n,
               'psi_lcn':units.V, 'psi_eq':units.V}

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
            the chosen parameter experiences large changes. The default is 'Eg'.
        step_min : number
            The smallest possible grid step. The default is 1e-7.
        step_max : number
            The largest possible grid step. The default is 20e-7.
        sigma : number
            Standard deviation of the gaussian function, which is used for
            smoothing the parameter change function. Roughly describes the size
            of area that is used to choose the local grid step. The default is
            100e-7.
        y0 : number or None
            The 'external' value of parameter 'param_grid' at x<0. Does not
            have any physical meaning, added in order to reduce the grid step
            at the boundary. `None` means there is no discontinuity in
            `param_grid` at x=0. The default is 0.
        yn : number or None
            `y0` analogue for the second boundary (x>x_max). The default is 0.
        """
        params = device.get_params()
        assert all([p in params for p in input_params])
        assert device.ready

        # creating 1D mesh
        print("Generating uniform mesh...", end=' ')
        self._gen_uni_mesh(device, step)
        print("complete (%d nodes)"%(len(self.x),))
        if not uniform:
            print("Generating non-uniform mesh...", end=' ')
            self._gen_nonuni_mesh(device, **options)
            print("complete (%d nodes)"%(len(self.x),))
        # mesh is stored in self.x
        self.xm = (self.x[1:]+self.x[:-1])/2  # midpoints

        # calculating physical parameters' values at mesh nodes
        self.values = dict()
        n = len(self.x)
        for p in input_params:
            self.values[p] = np.zeros(n)
            for i, xi in enumerate(self.x):
                self.values[p][i] = device.get_value(p, xi)
        # additional parameters and constants
        self.Vt = const.kb*const.T
        self.q = const.q
        self.eps_0 = const.eps_0
        self.values['Eg'] = self.values['Ec'] - self.values['Ev']
        self.values['C_dop'] = self.values['Nd'] - self.values['Na']
        self.values['ni'] = np.sqrt(self.values['Nc']*self.values['Nv']) \
                            * np.exp(-self.values['Eg'] / (2*self.Vt))

        # calculating selected phys. parameter's values at midpoints
        self.midp_values = dict()
        for p in midp_params:
            self.midp_values[p] = np.zeros(n-1)
            for i, xi in enumerate(self.xm):
                self.midp_values[p][i] = device.get_value(p, xi)

        # tracking measurement units and storing constants
        self.is_nondimensional = False
        self.solved_lcn = False
        self.solved_equilibrium = False

    def make_dimensionless(self):
        "Make every parameter dimensionless."
        assert not self.is_nondimensional

        # mesh
        self.x /= units.x
        self.xm /= units.x

        # parameters at mesh nodes and midpoints
        for p in self.values:
            self.values[p] /= unit_values[p]
        for p in self.midp_values:
            self.midp_values[p] /= unit_values[p]

        # constants
        self.Vt /= units.E
        self.q /= units.q
        self.eps_0 = 1.0

        self.is_nondimensional = True

    def original_units(self):
        "Return from dimensionless to original units."
        assert self.is_nondimensional

        # mesh
        self.x *= units.x
        self.xm *= units.x

        # parameters at mesh nodes and midpoints
        for p in self.values:
            self.values[p] *= unit_values[p]
        for p in self.midp_values:
            self.midp_values[p] *= unit_values[p]

        # constants
        self.Vt *= units.E
        self.q *= units.q
        self.eps_0 = const.eps_0

        self.is_nondimensional = False

    def solve_lcn(self, n_iter=20, lam=1.0, delta_max=1e-8):
        "Find built-in potential assuming local charge neutrality."
        C_dop = self.values['C_dop']
        Nc = self.values['Nc']
        Nv = self.values['Nv']
        Ec = self.values['Ec']
        Ev = self.values['Ev']
        Vt = self.Vt
        self.values['psi_lcn'] = eq.Ef_lcn_fermi(C_dop, Nc, Nv, Ec, Ev, Vt,
                                                 n_iter, lam, delta_max)
        self.solved_lcn = True

    def solve_equilibrium(self, n_iter=3000, lam=1.0, delta_max=1e-6):
        "Solve Poisson's equation at equilibrium."
        x = self.x
        xm = self.xm
        q = self.q
        eps_0 = self.eps_0
        eps = self.values['eps']
        C_dop = self.values['C_dop']
        Nc = self.values['Nc']
        Nv = self.values['Nv']
        Ec = self.values['Ec']
        Ev = self.values['Ev']
        Vt = self.Vt

        # psi_lcn -- initial guess for built-in potential
        if not self.solved_lcn:
            self.solve_lcn()
        psi = self.values['psi_lcn'].copy()

        # Newton's method
        self.delta = np.zeros(n_iter)  # change in psi
        for i in range(n_iter):
            A = eq.poisson_jac(psi, x, xm, eps, eps_0, q,
                               C_dop, Nc, Nv, Ec, Ev, Vt)
            b = eq.poisson_res(psi, x, xm, eps, eps_0, q,
                               C_dop, Nc, Nv, Ec, Ev, Vt)
            dpsi = solve_banded((1, 1), A, -b)
            self.delta[i] = np.mean(np.abs(dpsi))
            psi[1:-1] += lam*dpsi

        # storing solution and equilibrium concentrations
        # assert self.delta[-1]<delta_max
        self.values['psi_eq'] = psi
        self.values['n0'] = cc.n(psi, 0, Nc, Ec, Vt)
        self.values['p0']= cc.p(psi, 0, Nv, Ev, Vt)
        self.solved_equilibrium = True

if __name__ == '__main__':
    from sample_diode import sd
    import matplotlib.pyplot as plt

    # generating mesh
    dd = DiodeData(sd)
    x = dd.x
    Ec = dd.values['Ec']
    Ev = dd.values['Ev']
    mu_p = dd.values['mu_p']

    # plotting results
    plt.figure('Mesh generation')
    plt.plot(x, Ec, 'k-', lw=1.0)
    plt.plot(x, Ev, 'k-', lw=1.0)
    plt.xlabel('$x$')
    plt.ylabel('$E$')
    plt.twinx()
    plt.plot(x, mu_p, 'b:x', ms=4, lw=1.0)
    plt.ylabel('$\mu_p$', color='b')

    # solve Poisson's equation
    dd.make_dimensionless()  # does not converge with original units -- error?
    dd.solve_equilibrium()
    dd.original_units()
    psi = dd.values['psi_eq']
    n0 = dd.values['n0']
    p0 = dd.values['p0']

    # plot band diagram at equilibrium
    plt.figure('Equilibrium')
    plt.plot(x, Ec-psi, 'k-', lw=1.0)
    plt.plot(x, Ev-psi, 'k-', lw=1.0)
    plt.xlabel('$x$')
    plt.ylabel('$E$')
    plt.twinx()
    plt.plot(x, n0, 'b-', lw=0.5)
    plt.plot(x, p0, 'r-', lw=0.5)
    plt.ylabel('$n_0$, $p_0$')
    plt.yscale('log')

    plt.figure('Convergence')
    plt.plot(dd.delta)
    plt.xlabel('Iteration number')
    plt.yscale('log')
