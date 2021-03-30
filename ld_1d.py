# -*- coding: utf-8 -*-
"""
Class for a 1-dimensional model of a laser diode.
"""

import warnings
import numpy as np
from scipy.interpolate import interp1d
from slice_1d import Slice
import constants as const
import units

inp_params = ['Ev', 'Ec', 'Nd', 'Na', 'Nc', 'Nv', 'mu_n', 'mu_p', 'tau_n',
              'tau_p', 'B', 'Cn', 'Cp', 'eps', 'n_refr', 'g0', 'N_tr']
ybn_params = ['Ev', 'Ec', 'Nc', 'Nv', 'mu_n', 'mu_p']
unit_values = {'Ev':units.E, 'Ec':units.E, 'Eg':units.E, 'Nd':units.n,
               'Na':units.n, 'C_dop':units.n, 'Nc':units.n, 'Nv':units.n,
               'mu_n':units.mu, 'mu_p':units.mu, 'tau_n':units.t,
               'tau_p':units.t, 'B':1/(units.n*units.t), 
               'Cn':1/(units.n**2*units.t), 'Cp':1/(units.n**2*units.t),
               'eps':1, 'n_refr':1, 'ni':units.n, 'wg_mode':1/units.x,
               'n0':units.n, 'p0':units.n, 'psi_lcn':units.V, 'psi_eq':units.V,
               'g0':1/units.x, 'N_tr':units.n}

class LaserDiode1D(object):

    def __init__(self, slc, ar_inds, L, w, R1, R2, lam, ng, alpha_i, beta_sp):
        """
        Class for storing all the model parameters of a 1D laser diode.

        Parameters
        ----------
        slc : slice_1d.Slice
            A `Slice` object with all necessary physical parameters.
        ar_inds : number or list of numbers
            Indices of active region layers in `slc`.
        L : number
            Resonator length.
        w : number
            Stripe width.
        R1 : float
            Back (x=0) mirror reflectivity.
        R2 : float
            Front (x=L) mirror reflectivity..
        lam : number
            Operating wavelength.
        ng : number
            Group refractive index.
        alpha_i : number
            Internal optical loss (cm-1). Should not include free carrier
            absorption.
        beta_sp : number
            Spontaneous emission factor, i.e. the fraction of spontaneous
            emission that contributes to the lasing mode.

        """
        # checking if all the necessary parameters were specified
        # and if active region indices correspond to actual layers
        assert isinstance(slc, Slice)
        inds = list()
        for ind, layer in slc.layers.items():
            inds.append(ind)
            layer.check(inp_params)  # raises exception if fails
        if isinstance(ar_inds, int):
            self.ar_inds = [ar_inds]
        assert all([ar_ind in inds for ar_ind in self.ar_inds])
        self.slc = slc

        # device parameters
        self.L = L
        self.w = w
        self.R1 = R1
        self.R2 = R2
        assert all([r>0 and r<1 for r in (R1, R2)])
        self.alpha_m = 1/(2*L) * np.log(1/(R1*R2))
        self.lam = lam
        self.photon_energy = const.h*const.c/lam
        self.ng = ng
        self.vg = const.c/ng

        # parameters at mesh nodes
        self.yin = dict()  # values at interior nodes
        self.ybn = dict()  # values at boundary nodes
        self.sol = dict()  # current solution (potentials and concentrations)

    def gen_uniform_mesh(self, step=1e-7):
        """
        Generate a uniform mesh with a specified `step`. Should result with at
        least 50 nodes, otherwise raises an exception.
        """
        d = self.slc.get_thickness()
        if d/step < 50:
            raise Exception("Mesh step (%f) is too large." % step)
        self.xin = np.arange(0, d, step)
        self.xbn = (self.xin[1:]+self.xin[:-1]) / 2

    def calculate_param(self, p, nodes='internal'):
        """
        Calculate values of parameter `p` at all mesh nodes.

        Parameters
        ----------
        p : str
            Parameter name.
        nodes : str, optional
            Which mesh nodes -- `internal` (`i`) or `boundary` (`b`) -- should
            the values be calculated at. The default is 'internal'.

        Raises
        ------
        Exception
            If `p` is an unknown parameter name.

        """
        # checking if there is a mesh
        try:
            self.xin
        except AttributeError:
            warnings.warn('Warning: trying to calculate parameter %s before '
                          'generating mesh. Will use default uniform mesh.'%p)
            self.gen_uniform_mesh()
        # picking nodes
        if nodes=='internal' or nodes=='i':
            x = self.xin
            d = self.yin
        elif nodes=='boundary' or nodes=='b':
            x = self.xbn
            d = self.ybn

        # calculating values
        if p in inp_params:
            y = np.array([self.slc.get_value(p, xi) for xi in x])
        elif p=='Eg':
            Ec = np.array([self.slc.get_value('Ec', xi) for xi in x])
            Ev = np.array([self.slc.get_value('Ev', xi) for xi in x])
            y = Ec-Ev
        elif p=='C_dop':
            Nd = np.array([self.slc.get_value('Nd', xi) for xi in x])
            Na = np.array([self.slc.get_value('Na', xi) for xi in x])
            y = Nd-Na
        else:
            raise Exception('Error: unknown parameter %s' % p)
        d[p] = y  # modifies self.yin or self.ybn

    def gen_nonuniform_mesh(self, step_min=1e-7, step_max=20e-7, step_uni=5e-8,
                            param='Eg', sigma=100e-7, y_ext=[0, 0]):
        """
        Generate nonuniform mesh with step inversely proportional to change in
        `param` value. Uses gaussian function for smoothing.

        Parameters
        ----------
        step_min : float, optional
            Minimum step (cm). The default is 1e-7.
        step_max : float, optional
            Maximum step (cm). The default is 20e-7.
        step_uni : float, optional
            Step of the uniform mesh that is used to calculate `param` values.
                The default is 5e-8.
        param : str, optional
            Name of the parameter that is used to decide local step.
            The default is 'Eg'.
        sigma : float, optional
            Gaussian function standard deviation. The default is 100e-7.
        y_ext : TYPE, optional
            `param` values outside the laser boundaries. These are used for
            obtaining finer mesh at contacts. Pass [None, None] to disable
            this feature. The default is [0, 0].

        """
        def gauss(x, mu, sigma):
            return np.exp( -(x-mu)**2 / (2*sigma**2) )

        self.gen_uniform_mesh(step=step_uni)
        x = self.xin.copy()
        self.calculate_param(param, nodes='i')
        y = self.yin[param]

        # adding external values to y
        # same value as inside if y0 or yn is not a number
        if not isinstance(y_ext[0], (float, int)):
            y_ext[0] = y[0]
        if not isinstance(y_ext[-1], (float, int)):
            y_ext[1] = y[-1]
        y_ext = np.concatenate([ np.array([y_ext[0]]),
                                 y,
                                 np.array([y_ext[1]]) ])

        # function for choosing local step size
        f = np.abs(y_ext[2:]-y_ext[:-2])  # change of y at every point
        fg = np.zeros_like(f)  # convolution for smoothing
        for i, xi in enumerate(x):
            g = gauss(x, xi, sigma)
            fg[i] = np.sum(f*g)
        fg_fun = interp1d(x, fg/fg.max())

        # generating new grid
        k = step_max-step_min
        new_grid = list()
        xi = 0
        while xi<=x[-1]:
            new_grid.append(xi)
            xi += step_min + k*(1-fg_fun(xi))
        self.xin = np.array(new_grid)
        self.xbn = (self.xin[1:] + self.xin[:-1]) / 2

    def calc_all_params(self):
        "Calculate all parameters' values at mesh nodes."
        for p in inp_params+['Eg', 'C_dop']:
            self.calculate_param(p, 'i')
        for p in ybn_params:
            self.calculate_param(p, 'b')

#%%
if __name__=='__main__':
    import matplotlib.pyplot as plt
    from sample_slice import sl

    ld = LaserDiode1D(slc=sl, ar_inds=3,
                      L=3000e-4, w=100e-4,
                      R1=0.95, R2=0.05,
                      lam=0.87e-4, ng=3.9,
                      alpha_i=0.5, beta_sp=1e-4)
    ld.gen_nonuniform_mesh(param='Eg', y_ext=[0, 0])
    ld.calc_all_params()
    x = ld.xin*1e4
    plt.figure()
    plt.plot(x, ld.yin['Ec'], lw=0.5, color='b')
    plt.plot(x, ld.yin['Ev'], lw=0.5, color='b')
    plt.xlabel(r'$x$ ($\mu$m)')
    plt.ylabel('Energy (eV)', color='b')
    plt.twinx()
    plt.plot(x, ld.yin['n_refr'], lw=0.5, ls=':', color='g', marker='x',
             ms=3)
    plt.ylabel('Refractive index', color='g')
