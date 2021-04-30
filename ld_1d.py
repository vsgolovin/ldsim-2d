# -*- coding: utf-8 -*-
"""
Class for a 1-dimensional model of a laser diode.
"""

import warnings
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import solve_banded
from scipy import sparse
from slice_1d import Slice
import constants as const
import units
import carrier_concentrations as cc
import equilibrium as eq
import newton
import waveguide
import vrs
import flux
import recombination as rec

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
               'n':units.n, 'p':units.n,
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
        self.n_eff = None
        self.gamma = None

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
        self.npoints = len(self.xin)
        self.calc_all_params()

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
        self.npoints = len(self.xin)
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

    def calc_all_params(self):
        "Calculate all parameters' values at mesh nodes."
        for p in inp_params+['Eg', 'C_dop']:
            self.calculate_param(p, 'i')
        for p in ybn_params:
            self.calculate_param(p, 'b')
        if self.n_eff is not None:  # waveguide problem has been solved
            self._calc_wg_mode()

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
        self._update_solution(sol.x)

    def solve_waveguide(self, step=1e-7, n_modes=3, remove_layers=[0, 0]):
        ""
        # generating refractive index profile
        x = np.arange(0, self.slc.get_thickness(), step)
        n = np.array([self.slc.get_value('n_refr', xi) for xi in x])
        inds = np.array([self.slc.get_index(xi) for xi in x], dtype=int)

        # removing boundary layers (if needed)
        i1, i2 = remove_layers
        to_remove = self.slc.inds[:i1]
        if i2>0:
            to_remove += self.slc.inds[-i2:]
        ix = np.array([True]*len(inds))
        for layer_index in to_remove:
            ix = np.logical_and(ix, inds!=layer_index)
        x = x[ix]
        n = n[ix]
        inds = inds[ix]

        # active region location
        ar_ix = np.array([True]*len(inds))
        for layer_index in self.ar_inds:
            ar_ix = np.logical_and(ar_ix, inds==layer_index)

        # solving the eigenvalue problem
        n_eff_values, modes = waveguide.solve_wg(x, n, self.lam, n_modes)
        # and picking one mode with the largest confinement factor (Gamma)
        gammas = np.zeros(n_modes)
        for i in range(n_modes):
            mode = modes[:, i]
            gammas[i] = (mode*step)[ar_ix].sum()  # modes are normalized
        i = np.argmax(gammas)
        mode = modes[:, i]

        # storing results
        self.n_eff = n_eff_values[i]
        self.gamma = gammas[i]
        self.wgm_fun = interp1d(x, mode, bounds_error=False,
                                fill_value=0)
        self.wgm_fun_dls = interp1d(x/units.x, mode*units.x,
                                    bounds_error=False, fill_value=0)
        self._calc_wg_mode()

        return x, n, mode

    def _calc_wg_mode(self):
        "Calculate normalized laser mode profile and store in `self.yin`."
        assert self.n_eff is not None
        if self.is_dimensionless:
            self.yin['wg_mode'] = self.wgm_fun_dls(self.xin)
        else:
            self.yin['wg_mode'] = self.wgm_fun(self.xin)

    def _update_solution(self, v):
        "Update `self.sol` using calculated potentials vector `v`."
        if len(v)==3*(self.npoints-2):  # change in psi, phi_n, phi_p
            m = self.npoints-2
            self.sol['psi'][1:-1] += v[:m]
            self.sol['phi_n'][1:-1] += v[m:2*m]
            self.sol['phi_p'][1:-1] += v[2*m:]
        else:
            assert len(v)==self.npoints
            self.sol['psi'] = v.copy()
            self.sol['phi_n'] = np.zeros_like(v)
            self.sol['phi_p'] = np.zeros_like(v)
        self.sol['n'] = cc.n(self.sol['psi'], self.sol['phi_n'],
                             self.yin['Nc'], self.yin['Ec'], self.Vt)
        self.sol['p'] = cc.p(self.sol['psi'], self.sol['phi_p'],
                             self.yin['Nv'], self.yin['Ev'], self.Vt)

    def transport_init(self, voltage, psi_init=None, phi_n_init=None,
                       phi_p_init=None):
        "Initialize carrier transport problem at some external voltage."
        self.iterations = 0
        self.fluct = list()

        # copying passed initial guesses
        for arr, key in zip([psi_init, phi_n_init, phi_p_init],
                            ['psi', 'phi_n', 'phi_p']):
            if arr is not None:
                assert len(arr)==self.npoints
                self.sol[key] = arr.copy()

        # boundary conditions
        if self.is_dimensionless:
            voltage /= units.V
        self.sol['psi'][0] = self.yin['psi_bi'][0]-voltage/2
        self.sol['psi'][-1] = self.yin['psi_bi'][-1]+voltage/2
        self.sol['phi_n'][0] = -voltage/2
        self.sol['phi_n'][-1] = voltage/2
        self.sol['phi_p'][0] = -voltage/2
        self.sol['phi_p'][-1] = voltage/2

    def transport_step(self, omega=1.0):
        ""
        m = self.npoints-2
        h = self.xin[1:]-self.xin[:-1]  # npoints-1
        w = self.xbn[1:]-self.xbn[:-1]  # npoints-2 (m)
        psi = self.sol['psi']
        phi_n = self.sol['phi_n']
        phi_p = self.sol['phi_p']

        # carrier densities
        # indices 1 and 2 correspond to backward and forward values
        # for SG discretization
        n = self.sol['n']
        n1 = cc.n(psi[:-1], phi_n[:-1], self.ybn['Nc'], self.ybn['Ec'],
                  self.Vt)  # bacward calculation for SG flux
        n2 = cc.n(psi[1:], phi_n[1:], self.ybn['Nc'], self.ybn['Ec'],
                  self.Vt)  # forward calculation for SG flux
        p = self.sol['p']
        p1 = cc.p(psi[:-1], phi_p[:-1], self.ybn['Nv'], self.ybn['Ev'],
                  self.Vt)
        p2 = cc.p(psi[1:], phi_p[1:], self.ybn['Nv'], self.ybn['Ev'],
                  self.Vt)

        # current densities
        B_plus = flux.bernoulli(+(psi[1:]-psi[:-1])/self.Vt)   # npoints-1
        B_minus = flux.bernoulli(-(psi[1:]-psi[:-1])/self.Vt)
        jn = flux.SG_jn(n1, n2, B_plus, B_minus, h,
                        self.Vt, self.q, self.ybn['mu_n'])
        jp = flux.SG_jp(p1, p2, B_plus, B_minus, h,
                        self.Vt, self.q, self.ybn['mu_p'])

        # recombination rates
        R_srh = rec.srh_R(n[1:-1], p[1:-1],
                          self.yin['n0'][1:-1], self.yin['p0'][1:-1],
                          self.yin['tau_n'][1:-1], self.yin['tau_p'][1:-1])
        R_rad = rec.rad_R(n[1:-1], p[1:-1],
                          self.yin['n0'][1:-1], self.yin['p0'][1:-1],
                          self.yin['B'][1:-1])
        R_aug = rec.auger_R(n[1:-1], p[1:-1],
                            self.yin['n0'][1:-1], self.yin['p0'][1:-1],
                            self.yin['Cn'][1:-1], self.yin['Cp'][1:-1])
        R = R_srh + R_rad + R_aug

        # calculating residuals
        rvec = np.zeros(m*3)
        rvec[:m] = vrs.poisson_res(psi, n, p, h, w, self.yin['eps'],
                                   self.eps_0, self.q, self.yin['C_dop'])
        rvec[m:2*m] = -self.q*(-R)*w - (jn[1:]-jn[:-1])
        rvec[2*m:]  =  self.q*(-R)*w - (jp[1:]-jp[:-1])

        # carrier densities' derivatives
        dn_dpsi = cc.dn_dpsi(psi, phi_n, self.yin['Nc'],
                             self.yin['Ec'], self.Vt)
        dn_dphin = cc.dn_dphin(psi, phi_n, self.yin['Nc'],
                               self.yin['Ec'], self.Vt)
        dp_dpsi = cc.dp_dpsi(psi, phi_p, self.yin['Nv'],
                             self.yin['Ev'], self.Vt)
        dp_dphip = cc.dp_dphip(psi, phi_p, self.yin['Nv'],
                               self.yin['Ev'], self.Vt)

        # recombination rate derivatives
        dR_dpsi = (rec.srh_Rdot(n[1:-1], dn_dpsi[1:-1],
                                p[1:-1], dp_dpsi[1:-1],
                                self.yin['n0'][1:-1],
                                self.yin['p0'][1:-1],
                                self.yin['tau_n'][1:-1],
                                self.yin['tau_p'][1:-1])
                  +rec.rad_Rdot(n[1:-1], dn_dpsi[1:-1],
                                p[1:-1], dp_dpsi[1:-1], 
                                self.yin['n0'][1:-1],
                                self.yin['p0'][1:-1],
                                self.yin['B'][1:-1])
                  +rec.auger_Rdot(n[1:-1], dn_dpsi[1:-1],
                                  p[1:-1], dp_dpsi[1:-1],
                                  self.yin['n0'][1:-1],
                                  self.yin['p0'][1:-1],
                                  self.yin['Cn'][1:-1],
                                  self.yin['Cp'][1:-1]))
        dR_dphin = (rec.srh_Rdot(n[1:-1], dn_dphin[1:-1], p[1:-1], 0,
                                 self.yin['n0'][1:-1],
                                 self.yin['p0'][1:-1],
                                 self.yin['tau_n'][1:-1],
                                 self.yin['tau_p'][1:-1])
                   +rec.rad_Rdot(n[1:-1], dn_dphin[1:-1], p[1:-1], 0,
                                 self.yin['n0'][1:-1],
                                 self.yin['p0'][1:-1],
                                 self.yin['B'][1:-1])
                   +rec.auger_Rdot(n[1:-1], dn_dphin[1:-1], p[1:-1], 0,
                                   self.yin['n0'][1:-1],
                                   self.yin['p0'][1:-1],
                                   self.yin['Cn'][1:-1],
                                   self.yin['Cp'][1:-1]))
        dR_dphip = (rec.srh_Rdot(n[1:-1], 0, p[1:-1], dp_dphip[1:-1],
                                 self.yin['n0'][1:-1],
                                 self.yin['p0'][1:-1],
                                 self.yin['tau_n'][1:-1],
                                 self.yin['tau_p'][1:-1])
                   +rec.rad_Rdot(n[1:-1], 0, p[1:-1], dp_dphip[1:-1],
                                self.yin['n0'][1:-1], self.yin['p0'][1:-1],
                                self.yin['B'][1:-1])
                   +rec.auger_Rdot(n[1:-1], 0, p[1:-1], dp_dphip[1:-1],
                                   self.yin['n0'][1:-1],
                                   self.yin['p0'][1:-1],
                                   self.yin['Cn'][1:-1],
                                   self.yin['Cp'][1:-1]))

"""         # calculating Jacobian
        j11 = vrs.poisson_dF_dpsi(dn_dpsi, dp_dpsi, h, w, self.yin['eps'],
                                  self.eps_0, self.q)
        j12 = vrs.poisson_dF_dphin(dn_dphin, w, self.eps_0, self.q)
        j13 = vrs.poisson_dF_dphip(dp_dphip, w, self.eps_0, self.q)
        Bdot_plus = flux.bernoulli_dot(+(psi[1:]-psi[:-1])/self.Vt)  # m+1
        Bdot_minus = flux.bernoulli_dot(-(psi[1:]-psi[:-1])/self.Vt)
        djn_dpsi1 = flux.SG_djn_dpsi1(n, dn_dpsi, B_plus, B_minus,
                                      Bdot_plus, Bdot_minus, h, self.Vt,
                                      self.q, self.ybn['mu_n'])
        djn_dpsi2 = flux.SG_djn_dpsi2(n, dn_dpsi, B_plus, B_minus,
                                      Bdot_plus, Bdot_minus, h, self.Vt,
                                      self.q, self.ybn['mu_n'])
        djn_dphin1 = flux.SG_djn_dphin1(dn_dphin[:-1], B_minus, h, self.Vt,
                                        self.q, self.ybn['mu_n'])
        djn_dphin2 = flux.SG_djn_dphin2(dn_dphin[1:], B_plus, h, self.Vt,
                                        self.q, self.ybn['mu_n'])

        j11 = sparse.spdiags(j11, [1, 0, -1], m, m)
        j12 = sparse.spdiags(j12, [0,], m, m)
        j13 = sparse.spdiags(j13, [0,], m, m)
        J1 = sparse.hstack([j11, j12, j13])

        j21 = np.zeros((3, m))
        j21[0, 1:] = -djn_dpsi2[1:-1]
        j21[1, :] = self.q*dR_dpsi*w - (djn_dpsi1[1:]-djn_dpsi2[:-1])
        j21[2, :-1] = djn_dpsi1[1:-1]

        j22 = np.zeros((3, m))
        j22[0, 1:] = -djn_dphin2[1:-1]
        j22[1, :] = self.q*dR_dphin*w - (djn_dphin1[1:]-djn_dphin2[:-1])
        j22[2, :-1] = djn_dphin1[1:-1]

        j23 = np.zeros(m)
        j23[:] = self.q*dR_dphip*w

        j21 = sparse.spdiags(j21, [1, 0, -1], m, m)
        j22 = sparse.spdiags(j22, [1, 0, -1], m, m)
        j23 = sparse.spdiags(j23, [0,], m, m)
        J2 = sparse.hstack([j21, j22, j23])

        djp_dpsi1 = flux.SG_djp_dpsi1(p, dp_dpsi, B_plus, B_minus,
                                      Bdot_plus, Bdot_minus, h, self.Vt,
                                      self.q, self.ybn['mu_p'])
        djp_dpsi2 = flux.SG_djp_dpsi2(p, dp_dpsi, B_plus, B_minus,
                                      Bdot_plus, Bdot_minus, h, self.Vt,
                                      self.q, self.ybn['mu_p'])
        djp_dphip1 = flux.SG_djp_dphip1(dp_dphip[:-1], B_plus, h, self.Vt,
                                        self.q, self.ybn['mu_p'])
        djp_dphip2 = flux.SG_djp_dphip2(dp_dphip[1:], B_minus, h, self.Vt,
                                        self.q, self.ybn['mu_p'])

        j31 = np.zeros((3, m))
        j31[0, 1:] = -djp_dpsi2[1:-1]
        j31[1, :] = -self.q*dR_dpsi*w - (djp_dpsi1[1:]-djp_dpsi2[:-1])
        j31[2, :-1] = djp_dpsi1[1:-1]

        j32 = np.zeros(m)
        j32[:] = -self.q*dR_dphin*w

        j33 = np.zeros((3, m))
        j33[0, 1:] = -djp_dphip2[1:-1]
        j33[1, :] = -self.q*dR_dphip*w - (djp_dphip1[1:]-djp_dphip2[:-1])
        j33[2, :-1] = djp_dphip1[1:-1]

        j31 = sparse.spdiags(j31, [1, 0, -1], m, m)
        j32 = sparse.spdiags(j32, [0,], m, m)
        j33 = sparse.spdiags(j33, [1, 0, -1], m, m)
        J3 = sparse.hstack([j31, j32, j33])

        J = sparse.vstack([J1, J2, J3])
        J = J.tocsc()
        dx = sparse.linalg.spsolve(J, -rvec)
        self._update_solution(dx*omega) """

if __name__=='__main__':
    import matplotlib.pyplot as plt
    from sample_slice import sl

    plt.rc('lines', linewidth=0.7)
    plt.rc('figure.subplot', left=0.15, right=0.85)

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
    plt.plot(x, ld.yin['Ec'], color='b')
    plt.plot(x, ld.yin['Ev'], color='b')
    plt.xlabel(r'$x$ ($\mu$m)')
    plt.ylabel('Energy (eV)', color='b')
    plt.twinx()
    plt.plot(x, ld.yin['n_refr'], ls=':', color='g', marker='x',
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
    plt.plot(x, ld.yin['Ec']-psi, color='b')
    plt.plot(x, ld.yin['Ev']-psi, color='b')
    plt.xlabel(r'$x$ ($\mu$m)')
    plt.ylabel('Energy (eV)', color='b')
    plt.twinx()
    plt.plot(x, ld.sol['n'], color='g')
    plt.plot(x, ld.sol['p'], color='g', ls='--')
    plt.yscale('log')
    plt.ylabel('Carrier densities (cm$^{-3}$)', color='g')

    # 3. waveguide
    print('Calculating vertical mode profile...', end=' ')
    ld.solve_waveguide(remove_layers=[1, 1])
    print('Complete.')
    plt.figure('Waveguide')
    plt.plot(x, ld.yin['wg_mode']/1e4, color='b')
    plt.xlabel(r'$x$ ($\mu$m)')
    plt.ylabel('Mode intensity', color='b')
    plt.twinx()
    plt.plot(x, ld.yin['n_refr'], color='g')
    plt.ylabel('Refractive index', color='g')

    # 4. forward bias
    ld.make_dimensionless()
    ld.transport_init(0.1)
    for _ in range(20):
        ld.transport_step(1e-3)
    ld.original_units()
    plt.figure('Small forward bias')
    plt.plot(x, ld.yin['Ec']-ld.sol['psi'], color='b')
    plt.plot(x, ld.yin['Ev']-ld.sol['psi'], color='b')
    plt.plot(x, -ld.sol['phi_n'], color='g', ls=':')
    plt.plot(x, -ld.sol['phi_p'], color='r', ls=':')
    plt.xlabel(r'$x$ ($\mu$m)')
    plt.ylabel('Energy (eV)')
