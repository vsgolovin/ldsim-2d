# -*- coding: utf-8 -*-
"""
Class for storing all the parameters needed to simulate 1D semiconductor laser
operation.
"""

import numpy as np
from scipy.interpolate import interp1d
import units
import constants as const
from waveguide import solve_wg

# DiodeData-specific necessary input parameters
input_params = ['Ev', 'Ec', 'Nd', 'Na', 'Nc', 'Nv', 'mu_n', 'mu_p', 'tau_n',
                'tau_p', 'B', 'Cn', 'Cp', 'eps', 'n_refr']
midp_params = ['Ev', 'Ec', 'Nc', 'Nv', 'mu_n', 'mu_p']
unit_values = {'Ev':units.E, 'Ec':units.E, 'Eg':units.E, 'Nd':units.n,
               'Na':units.n, 'C_dop':units.n, 'Nc':units.n, 'Nv':units.n,
               'mu_n':units.mu, 'mu_p':units.mu, 'tau_n':units.t,
               'tau_p':units.t, 'B':1/(units.n*units.t), 
               'Cn':1/(units.n**2*units.t), 'Cp':1/(units.n**2*units.t),
               'eps':1, 'n_refr':1, 'ni':units.n, 'wg_mode':1/units.x}

class LaserData(object):

    def _calculate_values(self, device):
        self.values = dict()
        self.midp_values = dict()

        # input parameters at mesh nodes
        for p in input_params:
            self.values[p] = np.zeros_like(self.x)
            for i, xi in enumerate(self.x):
                self.values[p][i] = device.get_value(p, xi)
        # additional parameters
        self.values['Eg'] = self.values['Ec']-self.values['Ev']
        self.values['C_dop'] = self.values['Nd']-self.values['Na']
        self.values['ni'] = np.sqrt(self.values['Nc']*self.values['Nv']) \
                            * np.exp(-self.values['Eg'] / (2*self.Vt))

        # selected parameters at mesh midpoints
        for p in midp_params:
            self.midp_values[p] = np.zeros_like(self.xm)
            for i, xi in enumerate(self.xm):
                self.midp_values[p][i] = device.get_value(p, xi)

    def __init__(self, device, ar_ind, lam, L, w, R1, R2, alpha_i, beta_sp,
                 step=1e-7, n_modes=3, remove_cl=True):
        params = device.get_params()
        assert all([p in params for p in input_params])
        assert device.ready

        # laser properties
        self.ar_ind = ar_ind
        self.lam = lam
        self.L = L
        self.w = w
        self.R1 = R1
        self.R2 = R2
        self.alpha_i = alpha_i
        self.alpha_m = 1/(2*L) * np.log(1/(R1*R2))
        self.beta_sp = beta_sp

        # generating uniform mesh
        d = device.get_thickness()
        self.x = np.arange(0, d, step)
        self.xm = self.x[1:] - self.x[:-1]
        # boolean array for active region
        inds = np.array([device.get_index(xi) for xi in self.x])
        self.ar_ix = (inds==self.ar_ind)  # totally not confusing
        # constants and flags
        self.Vt = const.kb*const.T
        self.q = const.q
        self.eps_0 = const.eps_0
        self.is_dimensionless = False
        # calculating parameters' values
        self._calculate_values(device)

        # solving waveguide equation
        x = self.x.copy()
        n = self.values['n_refr']
        # removing narrow-gap contact layers
        if remove_cl:
            ind1 = device.inds[0]
            ind2 = device.inds[-1]
            ix = ~np.logical_or(inds==ind1, inds==ind2)
            x = x[ix]
            n = n[ix]
        n_eff, modes = solve_wg(x, n, lam, n_modes)

        # choosing mode with largest active region overlap
        inds = np.array([device.get_index(xi) for xi in x])
        ix = (inds==ar_ind)
        gammas = np.zeros(n_modes)
        for i in range(n_modes):
            mode = modes[:, i]
            gammas[i] = (mode*step)[ix].sum()
        i = np.argmax(gammas)
        self.n_eff = n_eff[i]
        self.values['wg_mode'] = modes[:, i]
        self.Gamma_f = interp1d(x, self.values['wg_mode'],
                                bounds_error=False, fill_value=0)
        self.Gamma_f_nd = interp1d(x/units.x, self.values['wg_mode']*units.x,
                                   bounds_error=False, fill_value=0)

    def generate_nonuniform_mesh(self, device, param='Eg', step_min=1e-7,
                                 step_max=20e-7, sigma=100e-7, y0=0, yn=0):
        def gauss(x, mu, sigma):
            return np.exp( -(x-mu)**2 / (2*sigma**2) )

        x = self.x.copy()
        y = self.values[param]

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
        self._calculate_values(device)
        if self.is_dimensionless:
            self.values['wg_mode'] = self.Gamma_f(self.x)
        else:
            self.values['wg_mode'] = self.Gamma_f_nd(self.x)

        # new boolean array for active region
        inds = np.array([device.get_index(xi) for xi in self.x])
        self.ar_ix = (inds==self.ar_ind)

    def make_dimensionless(self):
        "Make every parameter dimensionless."
        assert not self.is_dimensionless

        # mesh
        self.x /= units.x
        self.xm /= units.x

        # parameters at mesh nodes and midpoints
        for p in self.values:
            self.values[p] /= unit_values[p]
        for p in self.midp_values:
            self.midp_values[p] /= unit_values[p]

        # laser diode parameters
        self.L /= units.x
        self.w /= units.x
        self.alpha_i /= 1/units.x
        self.alpha_m /= 1/units.x

        # constants
        self.Vt /= units.E
        self.q /= units.q
        self.eps_0 = 1.0

        self.is_dimensionless = True

    def original_units(self):
        "Return from dimensionless to original units."
        assert self.is_dimensionless

        # mesh
        self.x *= units.x
        self.xm *= units.x

        # parameters at mesh nodes and midpoints
        for p in self.values:
            self.values[p] *= unit_values[p]
        for p in self.midp_values:
            self.midp_values[p] *= unit_values[p]

        # laser diode parameters
        self.L *= units.x
        self.w *= units.x
        self.alpha_i *= 1/units.x
        self.alpha_m *= 1/units.x

        # constants
        self.Vt *= units.E
        self.q *= units.q
        self.eps_0 = const.eps_0

        self.is_dimensionless = False


if __name__=='__main__':
    import matplotlib.pyplot as plt
    from sample_laser import sl
    ld = LaserData(sl, 3, 0.87e-4, 0.2, 0.01, 0.3, 0.3, 1.0, 1e-5)
    ld.generate_nonuniform_mesh(sl)
    ld.make_dimensionless()

    plt.figure("Sample laser / %d mesh points"%(len(ld.x)))
    plt.plot(ld.x, ld.Gamma_f_nd(ld.x), label=ld.n_eff, c='k')
    plt.xlabel('Coordinate')
    plt.ylabel('Vertical mode profile')
    plt.twinx()
    plt.plot(ld.x, ld.values['Eg'], c='b')
    plt.ylabel('Bandgap', c='b')
