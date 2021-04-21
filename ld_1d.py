# -*- coding: utf-8 -*-
"""
Class for a 1-dimensional model of a laser diode.
"""

import warnings
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import solve_banded
from slice_1d import Slice
import constants as const
import units
import carrier_concentrations as cc
import equilibrium as eq
import newton

inp_params = ['Ev', 'Ec', 'Nd', 'Na', 'Nc', 'Nv', 'mu_n', 'mu_p', 'tau_n',
              'tau_p', 'B', 'Cn', 'Cp', 'eps', 'n_refr', 'g0', 'N_tr']
ybn_params = ['Ev', 'Ec', 'Nc', 'Nv', 'mu_n', 'mu_p']
unit_values = {'Ev':units.E, 'Ec':units.E, 'Eg':units.E, 'Nd':units.n,
               'Na':units.n, 'C_dop':units.n, 'Nc':units.n, 'Nv':units.n,
               'mu_n':units.mu, 'mu_p':units.mu, 'tau_n':units.t,
               'tau_p':units.t, 'B':1/(units.n*units.t), 
               'Cn':1/(units.n**2*units.t), 'Cp':1/(units.n**2*units.t),
               'eps':1, 'n_refr':1, 'ni':units.n, 'wg_mode':1/units.x,
               'n0':units.n, 'p0':units.n, 'psi_lcn':units.V, 'psi_bi':units.V,
               'psi':units.V, 'phi_n':units.V, 'phi_p':units.V,
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

        # constants
        self.Vt = const.kb*const.T
        self.q = const.q
        self.eps_0 = const.eps_0
        self.fca_e = const.fca_e
        self.fca_h = const.fca_h

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

        self.gen_uniform_mesh()
        self.is_dimensionless = False

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
        self.calc_all_params()

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
        # picking nodes
        if nodes=='internal' or nodes=='i':
            x = self.xin
            d = self.yin
        elif nodes=='boundary' or nodes=='b':
            x = self.xbn
            d = self.ybn

        # calculating values
        dtype = np.dtype('float64')
        if p in inp_params:
            y = np.array([self.slc.get_value(p, xi) for xi in x], dtype=dtype)
        elif p=='Eg':
            Ec = np.array([self.slc.get_value('Ec', xi) for xi in x], dtype=dtype)
            Ev = np.array([self.slc.get_value('Ev', xi) for xi in x], dtype=dtype)
            y = Ec-Ev
        elif p=='C_dop':
            Nd = np.array([self.slc.get_value('Nd', xi) for xi in x], dtype=dtype)
            Na = np.array([self.slc.get_value('Na', xi) for xi in x], dtype=dtype)
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
        y_ext : number or NoneType, optional
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
        self.calc_all_params()

    def calc_all_params(self):
        "Calculate all parameters' values at mesh nodes."
        for p in inp_params+['Eg', 'C_dop']:
            self.calculate_param(p, 'i')
        for p in ybn_params:
            self.calculate_param(p, 'b')

    def make_dimensionless(self):
        "Make every parameter dimensionless."
        if self.is_dimensionless:
            return

        # constants
        self.Vt /= units.V
        self.q /= units.q
        self.eps_0 /= units.q/(units.x*units.V)
        self.fca_e /= 1/(units.n*units.x)
        self.fca_h /= 1/(units.n*units.x)

        # device parameters
        self.L /= units.x
        self.w /= units.x
        self.vg /= units.x/units.t
        # no need to convert other parameters (?)

        # mesh
        self.xin /= units.x
        self.xbn /= units.x

        # arrays
        for key in self.yin:
            self.yin[key] /= unit_values[key]
        for key in self.ybn:
            self.ybn[key] /= unit_values[key]
        for key in self.sol:
            self.sol[key] /= unit_values[key]

        self.is_dimensionless = True

    def original_units(self):
        "Convert all values back to original units."
        if not self.is_dimensionless:
            return

        # constants
        self.Vt *= units.V
        self.q *= units.q
        self.eps_0 *= units.q/(units.x*units.V)
        self.fca_e *= 1/(units.n*units.x)
        self.fca_h *= 1/(units.n*units.x)

        # device parameters
        self.L *= units.x
        self.w *= units.x
        self.vg *= units.x/units.t
        # no need to convert other parameters (?)

        # mesh
        self.xin *= units.x
        self.xbn *= units.x

        # arrays
        for key in self.yin:
            self.yin[key] *= unit_values[key]
        for key in self.ybn:
            self.ybn[key] *= unit_values[key]
        for key in self.sol:
            self.sol[key] *= unit_values[key]

        self.is_dimensionless = False

    def gen_lcn_solver(self):
        """
        Generate solver for electrostatic potential distribution along
        vertical axis at equilibrium assuming local charge neutrality.
        """
        def f(psi):
            n = cc.n(psi, 0, self.yin['Nc'], self.yin['Ec'], self.Vt)
            p = cc.p(psi, 0, self.yin['Nv'], self.yin['Ev'], self.Vt)
            return self.yin['C_dop']-n+p
        def fdot(psi):
            ndot = cc.dn_dpsi(psi, 0, self.yin['Nc'], self.yin['Ec'], self.Vt)
            pdot = cc.dp_dpsi(psi, 0, self.yin['Nv'], self.yin['Ev'], self.Vt)
            return -ndot+pdot

        # initial guess with Boltzmann statistics
        ni = eq.intrinsic_concentration(self.yin['Nc'], self.yin['Nv'],
                                        self.yin['Ec'], self.yin['Ev'], self.Vt)
        Ei = eq.intrinsic_level(self.yin['Nc'], self.yin['Nv'], self.yin['Ec'],
                                self.yin['Ev'], self.Vt)
        Ef_i = eq.Ef_lcn_boltzmann(self.yin['C_dop'], ni, Ei, self.Vt)

        # Jacobian is a vector -> element-wise division
        la_fun = lambda A, b: b/A

        sol = newton.NewtonSolver(f, fdot, Ef_i, la_fun)
        return sol

    def solve_lcn(self, maxiter=100, fluct=1e-8, omega=1):
        """
        Find potential distribution at zero external bias assuming local
        charge neutrality. Uses Newton's method implemented in `NewtonSolver`.

        Parameters
        ----------
        maxiter : int
            Maximum number of Newton's method iterations.
        fluct : float
            Fluctuation of solution that is needed to stop iterating before
            reaching `maxiter` steps.
        omega : float
            Damping parameter.

        """
        sol = self.gen_lcn_solver()
        sol.solve(maxiter, fluct, omega)
        if sol.fluct[-1]>fluct:
            warnings.warn('LaserDiode1D.solve_lcn(): fluctuation '+
                         ('%e exceeds %e.' % (sol.fluct[-1], fluct)))

        self.yin['psi_lcn'] = sol.x.copy()
        self.yin['n0'] = cc.n(psi=self.yin['psi_lcn'], phi_n=0,
                              Nc=self.yin['Nc'], Ec=self.yin['Ec'],
                              Vt=self.Vt)
        self.yin['p0'] = cc.p(psi=self.yin['psi_lcn'], phi_p=0,
                              Nv=self.yin['Nv'], Ev=self.yin['Ev'],
                              Vt=self.Vt)

    def gen_equilibrium_solver(self):
        """
        Generate solver for electrostatic potential distribution along
        vertical axis at equilibrium.
        """
        if 'psi_lcn' not in self.yin:
            self.solve_lcn()
        h = self.xin[1:]-self.xin[:-1]
        w = self.xbn[1:]-self.xbn[:-1]

        def res(psi):
            n = cc.n(psi, 0, self.yin['Nc'], self.yin['Ec'], self.Vt)
            p = cc.p(psi, 0, self.yin['Nv'], self.yin['Ev'], self.Vt)
            r = eq.poisson_res(psi, n, p, h, w, self.yin['eps'], self.eps_0,
                               self.q, self.yin['C_dop'])
            return r

        def jac(psi):
            n = cc.n(psi, 0, self.yin['Nc'], self.yin['Ec'], self.Vt)
            ndot = cc.dn_dpsi(psi, 0, self.yin['Nc'], self.yin['Ec'], self.Vt)
            p = cc.p(psi, 0, self.yin['Nv'], self.yin['Ev'], self.Vt)
            pdot = cc.dp_dpsi(psi, 0, self.yin['Nv'], self.yin['Ev'], self.Vt)
            j = eq.poisson_jac(psi, n, ndot, p, pdot, h, w, self.yin['eps'],
                                self.eps_0, self.q, self.yin['C_dop'])
            return j

        la_fun = lambda A, b: solve_banded((1,1), A, b)
        psi_init = self.yin['psi_lcn']
        sol = newton.NewtonSolver(res, jac, psi_init, la_fun,
                                  inds=np.arange(1, len(psi_init)-1))
        return sol

    def solve_equilibrium(self, maxiter=100, fluct=1e-8, omega=1):
        """
        Calculate electrostatic potential distribution at equilibrium (zero
        external bias). Uses Newton's method implemented in `NewtonSolver`.

        Parameters
        ----------
        maxiter : int
            Maximum number of Newton's method iterations.
        fluct : float
            Fluctuation of solution that is needed to stop iterating before
            reacing `maxiter` steps.
        omega : float
            Damping parameter.

        """
        sol = self.gen_equilibrium_solver()
        sol.solve(maxiter, fluct, omega)
        if sol.fluct[-1]>fluct:
            warnings.warn('LaserDiode1D.solve_equilibrium(): fluctuation '+
                         ('%e exceeds %e.' % (sol.fluct[-1], fluct)))
        self.yin['psi_bi'] = sol.x.copy()

#%%
if __name__=='__main__':
    import matplotlib.pyplot as plt
    from sample_slice import sl

    print('Creating an instance of LaserDiode1D...', end=' ')
    ld = LaserDiode1D(slc=sl, ar_inds=3,
                      L=3000e-4, w=100e-4,
                      R1=0.95, R2=0.05,
                      lam=0.87e-4, ng=3.9,
                      alpha_i=0.5, beta_sp=1e-4)
    print('Complete.')
    
    # 1. nonuniform mesh
    print('Generating a nonuniform mesh...', end=' ')
    ld.gen_nonuniform_mesh(param='Eg', y_ext=[0, 0])
    print('Complete.')
    x = ld.xin*1e4
    plt.figure('Flat bands')
    plt.plot(x, ld.yin['Ec'], lw=0.5, color='b')
    plt.plot(x, ld.yin['Ev'], lw=0.5, color='b')
    plt.xlabel(r'$x$ ($\mu$m)')
    plt.ylabel('Energy (eV)', color='b')
    plt.twinx()
    plt.plot(x, ld.yin['n_refr'], lw=0.5, ls=':', color='g', marker='x',
             ms=3)
    plt.ylabel('Refractive index', color='g')

    # 2. equilibrium
    ld.make_dimensionless()
    print('Calculating built-in potential assuming local charge neutrality...',
          end=' ')
    ld.solve_lcn()
    print('Complete.')
    print('Calculating built-in potential by solving Poisson\'s equation...',
          end=' ')
    ld.solve_equilibrium()
    print('Complete.')
    ld.original_units()
    psi = ld.yin['psi_bi']
    plt.figure('Equilibrium')
    plt.plot(x, ld.yin['Ec']-psi, lw=0.5, color='b')
    plt.plot(x, ld.yin['Ev']-psi, lw=0.5, color='b')
    plt.xlabel(r'$x$ ($\mu$m)')
    plt.ylabel('Energy (eV)')
