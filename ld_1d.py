# -*- coding: utf-8 -*-
"""
Class for a 1-dimensional model of a laser diode.
"""

import warnings
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import solve_banded
from scipy import sparse
from design_1d import Design1D
import constants as const
import units
import carrier_concentrations as cc
import equilibrium as eq
import newton
import waveguide
import vrs
import flux
import recombination as rec
import sdf

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
               'g0':1/units.x, 'N_tr':units.n, 'S':units.n, 'J':units.j}


class LaserDiode1D(object):

    def __init__(self, design, ar_inds, L, w, R1, R2,
                 lam, ng, alpha_i, beta_sp):
        """
        Class for storing all the model parameters of a 1D laser diode.

        Parameters
        ----------
        design : design_1d.Design1D
            A `Design_1D` object with all necessary physical parameters.
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
        assert isinstance(design, Design1D)
        inds = list()
        for ind, layer in design.layers.items():
            inds.append(ind)
            layer.check(inp_params)  # raises exception if fails
        if isinstance(ar_inds, int):
            self.ar_inds = [ar_inds]
        assert all([ar_ind in inds for ar_ind in self.ar_inds])
        self.design = design

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
        self.alpha_i = alpha_i
        self.beta_sp = beta_sp

        # parameters at mesh nodes
        self.yin = dict()  # values at interior nodes
        self.ybn = dict()  # values at boundary nodes
        self.sol = dict()  # current solution (potentials and concentrations)
        self.sol['S'] = 1e-12  # essentially 0

        self.gen_uniform_mesh()
        self.is_dimensionless = False

    def gen_uniform_mesh(self, step=1e-7):
        """
        Generate a uniform mesh with a specified `step`. Should result with at
        least 50 nodes, otherwise raises an exception.
        """
        d = self.design.get_thickness()
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
            The default is `5e-8`.
        param : str, optional
            Name of the parameter that is used to decide local step.
            The default is `'Eg'`.
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
            y = np.array([self.design.get_value(p, xi) for xi in x],
                         dtype=dtype)
        elif p=='Eg':
            Ec = np.array([self.design.get_value('Ec', xi) for xi in x],
                          dtype=dtype)
            Ev = np.array([self.design.get_value('Ev', xi) for xi in x],
                          dtype=dtype)
            y = Ec-Ev
        elif p=='C_dop':
            Nd = np.array([self.design.get_value('Nd', xi) for xi in x],
                          dtype=dtype)
            Na = np.array([self.design.get_value('Na', xi) for xi in x],
                          dtype=dtype)
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
        inds = np.array([self.design.get_index(xi) for xi in self.xin])
        self.ar_ix = np.zeros(self.xin.shape, dtype=bool)
        for ind in self.ar_inds:
            self.ar_ix |= (inds == ind)

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
        self.alpha_m /= 1/units.x
        self.alpha_i /= 1/units.x
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
        self.alpha_m *= 1/units.x
        self.alpha_i *= 1/units.x
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

    def solve_lcn(self, maxiter=100, fluct=1e-8, omega=1.0):
        """
        Find potential distribution at zero external bias assuming local
        charge neutrality. Uses Newton's method implemented in `NewtonSolver`.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of Newton's method iterations.
        fluct : float, optional
            Fluctuation of solution that is needed to stop iterating before
            reaching `maxiter` steps.
        omega : float, optional
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

    def solve_equilibrium(self, maxiter=100, fluct=1e-8, omega=1.0):
        """
        Calculate electrostatic potential distribution at equilibrium (zero
        external bias). Uses Newton's method implemented in `NewtonSolver`.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of Newton's method iterations.
        fluct : float, optional
            Fluctuation of solution that is needed to stop iterating before
            reacing `maxiter` steps.
        omega : float, optional
            Damping parameter.

        """
        sol = self.gen_equilibrium_solver()
        sol.solve(maxiter, fluct, omega)
        if sol.fluct[-1]>fluct:
            warnings.warn('LaserDiode1D.solve_equilibrium(): fluctuation '+
                         ('%e exceeds %e.' % (sol.fluct[-1], fluct)))
        self.yin['psi_bi'] = sol.x.copy()
        self.sol['psi'] = sol.x.copy()
        self.sol['phi_n'] = np.zeros_like(sol.x)
        self.sol['phi_p'] = np.zeros_like(sol.x)
        self._update_densities()

    def _update_densities(self):
        """
        Update electron and hole densities using currently stored potentials.
        """
        self.sol['n'] = cc.n(self.sol['psi'], self.sol['phi_n'],
                             self.yin['Nc'], self.yin['Ec'], self.Vt)
        self.sol['p'] = cc.p(self.sol['psi'], self.sol['phi_p'],
                             self.yin['Nv'], self.yin['Ev'], self.Vt)

    def solve_waveguide(self, step=1e-7, n_modes=3, remove_layers=(0, 0)):
        """
        Calculate vertical mode profile. Finds `n_modes` solutions of the
        eigenvalue problem with the highest eigenvalues (effective
        indices) and picks the one with the highest optical confinement
        factor (active region overlap).

        Parameters
        ----------
        step : float, optional
            Uniform mesh step (cm).
        n_modes : int, optional
            Number of calculated eigenproblem solutions.
        remove_layers : (int, int), optional
            Number of layers to exclude from calculated refractive index
            profile at each side of the device. Useful to exclude contact
            layers.

        """
        # generating refractive index profile
        x = np.arange(0, self.design.get_thickness(), step)
        n = np.array([self.design.get_value('n_refr', xi) for xi in x])
        inds = np.array([self.design.get_index(xi) for xi in x], dtype=int)

        # removing boundary layers (if needed)
        i1, i2 = remove_layers
        to_remove = self.design.inds[:i1]
        if i2 > 0:
            to_remove += self.design.inds[-i2:]
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

    def transport_init(self, voltage, psi_init=None, phi_n_init=None,
                       phi_p_init=None):
        "Initialize diode drift-diffusion problem."
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

    def _jn_SG(self, B_plus, B_minus, Bdot_plus, Bdot_minus, h):

        psi = self.sol['psi']
        phi_n = self.sol['phi_n']

        # electron densities at finite volume boundaries (m+1)
        n1 = cc.n(psi[:-1], phi_n[:-1], self.ybn['Nc'], self.ybn['Ec'],
                  self.Vt)
        n2 = cc.n(psi[1:], phi_n[1:], self.ybn['Nc'], self.ybn['Ec'],
                  self.Vt)

        # forward (2-2) and backward (1-1) derivatives
        # w.r.t. volume boundaries (m+1)
        dn1_dpsi1 = cc.dn_dpsi(psi[:-1], phi_n[:-1], self.ybn['Nc'],
                               self.ybn['Ec'], self.Vt)
        dn2_dpsi2 = cc.dn_dpsi(psi[1:], phi_n[1:], self.ybn['Nc'],
                               self.ybn['Ec'], self.Vt)
        dn1_dphin1 = cc.dn_dphin(psi[:-1], phi_n[:-1], self.ybn['Nc'],
                                 self.ybn['Ec'], self.Vt)
        dn2_dphin2 = cc.dn_dphin(psi[1:], phi_n[1:], self.ybn['Nc'],
                                 self.ybn['Ec'], self.Vt)

        # current densities and their derivatives (m+1)
        jn = flux.SG_jn(n1, n2, B_plus, B_minus, h,
                        self.Vt, self.q, self.ybn['mu_n'])
        djn_dpsi1 = flux.SG_djn_dpsi1(n1, n2, dn1_dpsi1, B_minus,
                                      Bdot_plus, Bdot_minus, h, self.Vt,
                                      self.q, self.ybn['mu_n'])
        djn_dpsi2 = flux.SG_djn_dpsi2(n1, n2, dn2_dpsi2, B_plus,
                                      Bdot_plus, Bdot_minus, h, self.Vt,
                                      self.q, self.ybn['mu_n'])
        djn_dphin1 = flux.SG_djn_dphin1(dn1_dphin1, B_minus, h, self.Vt,
                                        self.q, self.ybn['mu_n'])
        djn_dphin2 = flux.SG_djn_dphin2(dn2_dphin2, B_plus, h, self.Vt,
                                        self.q, self.ybn['mu_n'])

        return jn, djn_dpsi1, djn_dpsi2, djn_dphin1, djn_dphin2

    def _jp_SG(self, B_plus, B_minus, Bdot_plus, Bdot_minus, h):

        psi = self.sol['psi']
        phi_p = self.sol['phi_p']

        # hole densities at finite volume boundaries (m+1)
        p1 = cc.p(psi[:-1], phi_p[:-1], self.ybn['Nv'], self.ybn['Ev'],
                  self.Vt)
        p2 = cc.p(psi[1:], phi_p[1:], self.ybn['Nv'], self.ybn['Ev'],
                  self.Vt)

        # forward (2-2) and backward (1-1) derivatives
        # w.r.t. volume boundaries (m+1)
        dp1_dpsi1 = cc.dp_dpsi(psi[:-1], phi_p[:-1], self.ybn['Nv'],
                               self.ybn['Ev'], self.Vt)
        dp2_dpsi2 = cc.dp_dpsi(psi[1:], phi_p[1:], self.ybn['Nv'],
                               self.ybn['Ev'], self.Vt)
        dp1_dphip1 = cc.dp_dphip(psi[:-1], phi_p[:-1], self.ybn['Nv'],
                                 self.ybn['Ev'], self.Vt)
        dp2_dphip2 = cc.dp_dphip(psi[1:], phi_p[1:], self.ybn['Nv'],
                                 self.ybn['Ev'], self.Vt)

        # current densities and their derivatives (m+1)
        jp = flux.SG_jp(p1, p2, B_plus, B_minus, h,
                        self.Vt, self.q, self.ybn['mu_p'])
        djp_dpsi1 = flux.SG_djp_dpsi1(p1, p2, dp1_dpsi1, B_plus,
                                      Bdot_plus, Bdot_minus, h, self.Vt,
                                      self.q, self.ybn['mu_p'])
        djp_dpsi2 = flux.SG_djp_dpsi2(p1, p2, dp2_dpsi2, B_minus,
                                      Bdot_plus, Bdot_minus, h, self.Vt,
                                      self.q, self.ybn['mu_p'])
        djp_dphip1 = flux.SG_djp_dphip1(dp1_dphip1, B_plus, h, self.Vt,
                                        self.q, self.ybn['mu_p'])
        djp_dphip2 = flux.SG_djp_dphip2(dp2_dphip2, B_minus, h, self.Vt,
                                        self.q, self.ybn['mu_p'])

        return jp, djp_dpsi1, djp_dpsi2, djp_dphip1, djp_dphip2

    def _jn_mSG(self, B_plus, B_minus, Bdot_plus, Bdot_minus, h):

        psi = self.sol['psi']
        phi_n = self.sol['phi_n']

        # n = Nc * F(nu_n)
        F = sdf.fermi_fdint
        nu_n1 = (psi[:-1]-phi_n[:-1]-self.ybn['Ec']) / self.Vt  # (m+1)
        nu_n2 = (psi[1:]-phi_n[1:]-self.ybn['Ec']) / self.Vt
        exp_nu_n1 = np.exp(nu_n1)
        exp_nu_n2 = np.exp(nu_n2)

        # electron current density (m+1)
        gn = flux.g(nu_n1, nu_n2, F)
        jn_SG = flux.oSG_jn(exp_nu_n1, exp_nu_n2, B_plus, B_minus,
                            h, self.ybn['Nc'], self.Vt, self.q,
                            self.ybn['mu_n'])
        jn = jn_SG * gn

        # electron current density derivatives (m+1)
        Fdot = sdf.fermi_dot_fdint
        gdot_n1 = flux.gdot(gn, nu_n1, F, Fdot) / self.Vt
        gdot_n2 = flux.gdot(gn, nu_n2, F, Fdot) / self.Vt
        djn_dpsi1_SG = flux.oSG_djn_dpsi1(exp_nu_n1, exp_nu_n2,
                                          B_minus, Bdot_plus, Bdot_minus,
                                          h, self.ybn['Nc'], self.q,
                                          self.ybn['mu_n'])
        djn_dpsi1 = flux.mSG_jdot(jn_SG, djn_dpsi1_SG, gn, gdot_n1)
        djn_dpsi2_SG = flux.oSG_djn_dpsi2(exp_nu_n1, exp_nu_n2,
                                          B_plus, Bdot_plus, Bdot_minus,
                                          h, self.ybn['Nc'], self.q,
                                          self.ybn['mu_n'])
        djn_dpsi2 = flux.mSG_jdot(jn_SG, djn_dpsi2_SG, gn, gdot_n2)
        djn_dphin1_SG = flux.oSG_djn_dphin1(exp_nu_n1, B_minus, h,
                                            self.ybn['Nc'], self.q,
                                            self.ybn['mu_n'])
        djn_dphin1 = flux.mSG_jdot(jn_SG, djn_dphin1_SG, gn, -gdot_n1)
        djn_dphin2_SG = flux.oSG_djn_dphin2(exp_nu_n2, B_plus, h,
                                            self.ybn['Nc'], self.q,
                                            self.ybn['mu_n'])
        djn_dphin2 = flux.mSG_jdot(jn_SG, djn_dphin2_SG, gn, -gdot_n2)

        return jn, djn_dpsi1, djn_dpsi2, djn_dphin1, djn_dphin2

    def _jp_mSG(self, B_plus, B_minus, Bdot_plus, Bdot_minus, h):

        psi = self.sol['psi']
        phi_p = self.sol['phi_p']

        #  p = Nv * F(nu_p)
        F = sdf.fermi_fdint
        nu_p1 = (-psi[:-1]+phi_p[:-1]+self.ybn['Ev']) / self.Vt
        nu_p2 = (-psi[1:]+phi_p[1:]+self.ybn['Ev']) / self.Vt
        exp_nu_p1 = np.exp(nu_p1)
        exp_nu_p2 = np.exp(nu_p2)

        # hole current density (m+1)
        gp = flux.g(nu_p1, nu_p2, F)
        jp_SG = flux.oSG_jp(exp_nu_p1, exp_nu_p2, B_plus, B_minus,
                            h, self.ybn['Nv'], self.Vt, self.q,
                            self.ybn['mu_p'])
        jp = jp_SG * gp

        # hole current density derivatives (m+1)
        Fdot = sdf.fermi_dot_fdint
        gdot_p1 = flux.gdot(gp, nu_p1, F, Fdot) / self.Vt
        gdot_p2 = flux.gdot(gp, nu_p2, F, Fdot) / self.Vt
        djp_dpsi1_SG = flux.oSG_djp_dpsi1(exp_nu_p1, exp_nu_p2,
                                          B_plus, Bdot_plus, Bdot_minus,
                                          h, self.ybn['Nv'], self.q,
                                          self.ybn['mu_p'])
        djp_dpsi1 = flux.mSG_jdot(jp_SG, djp_dpsi1_SG, gp, -gdot_p1)
        djp_dpsi2_SG = flux.oSG_djp_dpsi2(exp_nu_p1, exp_nu_p2,
                                          B_minus, Bdot_plus, Bdot_minus,
                                          h, self.ybn['Nv'], self.q,
                                          self.ybn['mu_p'])
        djp_dpsi2 = flux.mSG_jdot(jp_SG, djp_dpsi2_SG, gp, -gdot_p2)
        djp_dphip1_SG = flux.oSG_djp_dphip1(exp_nu_p1, B_plus, h,
                                            self.ybn['Nv'], self.q,
                                            self.ybn['mu_p'])
        djp_dphip1 = flux.mSG_jdot(jp_SG, djp_dphip1_SG, gp, gdot_p1)
        djp_dphip2_SG = flux.oSG_djp_dphip2(exp_nu_p2, B_minus, h,
                                            self.ybn['Nv'], self.q,
                                            self.ybn['mu_p'])
        djp_dphip2 = flux.mSG_jdot(jp_SG, djp_dphip2_SG, gp, gdot_p2)

        return jp, djp_dpsi1, djp_dpsi2, djp_dphip1, djp_dphip2

    def _transport_system(self, discr='mSG'):
        """
        Calculate Jacobian and residual for the transport problem.

        Parameters
        ----------
        discr : str
            Current density discretiztion scheme.

        Returns
        -------
        J : scipy.sparse.cscmatrix
            Transport system Jacobian.
            Shape is `(self.npoints - 2, self.npoints - 2)`.
        r : numpy.ndarray
            Residual vector with length `self.npoints - 2`.

        """
        # mesh parameters
        m = self.npoints - 2  # number of inner nodes
                              # used in comments to show array shape
        h = self.xin[1:] - self.xin[:-1]  # mesh steps (m+1)
        w = self.xbn[1:] - self.xbn[:-1]  # volumes (m)

        # potentials, carrier densities and their derivatives at nodes
        psi = self.sol['psi']
        phi_n = self.sol['phi_n']
        phi_p = self.sol['phi_p']
        n = self.sol['n']
        p = self.sol['p']
        dn_dpsi = cc.dn_dpsi(psi, phi_n, self.yin['Nc'],
                             self.yin['Ec'], self.Vt)
        dn_dphin = cc.dn_dphin(psi, phi_n, self.yin['Nc'],
                               self.yin['Ec'], self.Vt)
        dp_dpsi = cc.dp_dpsi(psi, phi_p, self.yin['Nv'],
                             self.yin['Ev'], self.Vt)
        dp_dphip = cc.dp_dphip(psi, phi_p, self.yin['Nv'],
                               self.yin['Ev'], self.Vt)

        # Bernoulli function for current density calculation (m+1)
        B_plus = flux.bernoulli(+(psi[1:]-psi[:-1])/self.Vt)
        B_minus = flux.bernoulli(-(psi[1:]-psi[:-1])/self.Vt)
        Bdot_plus = flux.bernoulli_dot(+(psi[1:]-psi[:-1])/self.Vt)
        Bdot_minus = flux.bernoulli_dot(-(psi[1:]-psi[:-1])/self.Vt)

        # calculating current densities and their derivatives
        if discr == 'SG':  # Scharfetter-Gummel discretization
            jn, djn_dpsi1, djn_dpsi2, djn_dphin1, djn_dphin2 = \
                self._jn_SG(B_plus, B_minus, Bdot_plus, Bdot_minus, h)
            jp, djp_dpsi1, djp_dpsi2, djp_dphip1, djp_dphip2 = \
                self._jp_SG(B_plus, B_minus, Bdot_plus, Bdot_minus, h)

        elif discr == 'mSG':  # modified SG discretization
            jn, djn_dpsi1, djn_dpsi2, djn_dphin1, djn_dphin2 = \
                self._jn_mSG(B_plus, B_minus, Bdot_plus, Bdot_minus, h)
            jp, djp_dpsi1, djp_dpsi2, djp_dphip1, djp_dphip2 = \
                self._jp_mSG(B_plus, B_minus, Bdot_plus, Bdot_minus, h)

        else:
            raise Exception('Error: unknown current density '
                            + 'discretization scheme %s.' % discr)

        # store calculated current density
        self.sol['J'] = (jn + jp).mean()

        # recombination rates (m)
        n0 = self.yin['n0'][1:-1]
        p0 = self.yin['p0'][1:-1]
        tau_n = self.yin['tau_n'][1:-1]
        tau_p = self.yin['tau_p'][1:-1]
        B_rad = self.yin['B'][1:-1]
        Cn = self.yin['Cn'][1:-1]
        Cp = self.yin['Cp'][1:-1]
        R_srh = rec.srh_R(n[1:-1], p[1:-1], n0, p0, tau_n, tau_p)
        R_rad = rec.rad_R(n[1:-1], p[1:-1], n0, p0, B_rad)
        R_aug = rec.auger_R(n[1:-1], p[1:-1], n0, p0, Cn, Cp)
        R = (R_srh + R_rad + R_aug)

        # recombination rates' derivatives
        dR_dpsi = (rec.srh_Rdot(n[1:-1], dn_dpsi[1:-1],
                                p[1:-1], dp_dpsi[1:-1],
                                n0, p0, tau_n, tau_p)
                  +rec.rad_Rdot(n[1:-1], dn_dpsi[1:-1],
                                p[1:-1], dp_dpsi[1:-1],
                                n0, p0, B_rad)
                  +rec.auger_Rdot(n[1:-1], dn_dpsi[1:-1],
                                  p[1:-1], dp_dpsi[1:-1],
                                  n0, p0, Cn, Cp))
        dR_dphin = (rec.srh_Rdot(n[1:-1], dn_dphin[1:-1], p[1:-1], 0,
                                 n0, p0, tau_n, tau_p)
                   +rec.rad_Rdot(n[1:-1], dn_dphin[1:-1], p[1:-1], 0,
                                 n0, p0, B_rad)
                   +rec.auger_Rdot(n[1:-1], dn_dphin[1:-1], p[1:-1], 0,
                                   n0, p0, Cn, Cp))
        dR_dphip = (rec.srh_Rdot(n[1:-1], 0, p[1:-1], dp_dphip[1:-1],
                                 n0, p0, tau_n, tau_p)
                   +rec.rad_Rdot(n[1:-1], 0, p[1:-1], dp_dphip[1:-1],
                                 n0, p0, B_rad)
                   +rec.auger_Rdot(n[1:-1], 0, p[1:-1], dp_dphip[1:-1],
                                   n0, p0, Cn, Cp))

        # calculating residual of the system (m*3)
        rvec = np.zeros(m*3)
        rvec[:m] = vrs.poisson_res(psi, n, p, h, w, self.yin['eps'],
                                   self.eps_0, self.q, self.yin['C_dop'])
        rvec[m:2*m] =  self.q*R*w - (jn[1:]-jn[:-1])
        rvec[2*m:]  = -self.q*R*w - (jp[1:]-jp[:-1])

        # calculating Jacobian (m*3, m*3)
        # 1. Poisson's equation
        j11 = vrs.poisson_dF_dpsi(dn_dpsi, dp_dpsi, h, w, self.yin['eps'],
                                  self.eps_0, self.q)
        j12 = vrs.poisson_dF_dphin(dn_dphin, w, self.eps_0, self.q)
        j13 = vrs.poisson_dF_dphip(dp_dphip, w, self.eps_0, self.q)

        # 2. Electron current continuity equation
        j21 = vrs.jn_dF_dpsi(djn_dpsi1, djn_dpsi2, dR_dpsi, w, self.q, m)
        j22 = vrs.jn_dF_dphin(djn_dphin1, djn_dphin2, dR_dphin,
                              w, self.q, m)
        j23 = vrs.jn_dF_dphip(dR_dphip, w, self.q, m)

        # 3. Hole current continuity equation
        j31 = vrs.jp_dF_dpsi(djp_dpsi1, djp_dpsi2, dR_dpsi, w, self.q, m)
        j32 = vrs.jp_dF_dphin(dR_dphin, w, self.q, m)
        j33 = vrs.jp_dF_dphip(djp_dphip1, djp_dphip2, dR_dphip,
                              w, self.q, m)

        # collect Jacobian diagonals
        data = np.zeros((11, 3*m))
        data[0, 2*m:   ] = j13
        data[1,   m:2*m] = j12
        data[1, 2*m:   ] = j23
        data[2,    :m  ] = j11[0]
        data[2,   m:2*m] = j22[0]
        data[2, 2*m:   ] = j33[0]
        data[3,    :m  ] = j11[1]
        data[3,   m:2*m] = j22[1]
        data[3, 2*m:   ] = j33[1]
        data[4,    :m  ] = j11[2]
        data[4,   m:2*m] = j22[2]
        data[4, 2*m:   ] = j33[2]
        data[5,    :m  ] = j21[0]
        data[6,    :m  ] = j21[1]
        data[6,   m:2*m] = j32
        data[7,    :m  ] = j21[2]
        data[8,    :m  ] = j31[0]
        data[9,    :m  ] = j31[1]
        data[10,   :m  ] = j31[2]

        # assemble sparse matrix
        diags = [2*m, m, 1, 0, -1, -m+1, -m, -m-1, -2*m+1, -2*m, -2*m-1]
        J = sparse.spdiags(data=data, diags=diags, m=3*m, n=3*m, format='csc')

        return J, rvec

    def transport_step(self, omega=1.0, discr='mSG'):
        """
        Perform a single Newton step for the transport problem.

        Parameters
        ----------
        omega : float
            Damping parameter (`x += dx*omega`).
        discr : str
            Current density discretiztion scheme.

        """
        J, rvec = self._transport_system(discr)
        dx = sparse.linalg.spsolve(J, -rvec)

        # calculating and saving fluctuation
        x = np.hstack((self.sol['psi'][1:-1],
                       self.sol['phi_n'][1:-1],
                       self.sol['phi_p'][1:-1]))
        fluct = newton.l2_norm(dx) / newton.l2_norm(x)
        self.fluct.append(fluct)

        # updating current solution (potentials and densities)
        m = self.npoints - 2
        self.sol['psi'][1:-1] += dx[:m] * omega
        self.sol['phi_n'][1:-1] += dx[m:2*m] * omega
        self.sol['phi_p'][1:-1] += dx[2*m:] * omega
        self._update_densities()
        return fluct

    def _calculate_fca(self):
        "Calculate free-carrier absorption."
        T = self.yin['wg_mode'][1:-1]
        w = self.xbn[1:] - self.xbn[:-1]
        n = self.sol['n'][1:-1]
        p = self.sol['p'][1:-1]
        arr = T*w*(n*self.fca_e + p*self.fca_h)
        return np.sum(arr)

    def lasing_init(self, voltage, psi_init=None, phi_n_init=None,
                    phi_p_init=None, S_init=None):
        "Initialize laser diode drift-diffusion problem."
        self.transport_init(voltage, psi_init, phi_n_init, phi_p_init)
        if S_init is not None:
            assert isinstance(S_init, (float, int))
            self.sol['S'] = S_init

    def _lasing_system(self, discr='mSG'):
        """
        Calculate Jacobian and residual for the lasing problem.

        Parameters
        ----------
        discr : str
            Current density discretiztion scheme.

        Returns
        -------
        J : scipy.sparse.cscmatrix
            Lasing system Jacobian.
            Shape is `(self.npoints - 2, self.npoints - 2)`.
        r : numpy.ndarray
            Residual vector with length `self.npoints - 2`.

        """
        # mesh parameters
        m = self.npoints - 2  # number of inner nodes
                              # used in comments to show array shape
        h = self.xin[1:] - self.xin[:-1]  # mesh steps (m+1)
        w = self.xbn[1:] - self.xbn[:-1]  # volumes (m)

        # potentials, carrier densities and their derivatives at nodes
        psi = self.sol['psi']
        phi_n = self.sol['phi_n']
        phi_p = self.sol['phi_p']
        n = self.sol['n']
        p = self.sol['p']
        dn_dpsi = cc.dn_dpsi(psi, phi_n, self.yin['Nc'],
                             self.yin['Ec'], self.Vt)
        dn_dphin = cc.dn_dphin(psi, phi_n, self.yin['Nc'],
                               self.yin['Ec'], self.Vt)
        dp_dpsi = cc.dp_dpsi(psi, phi_p, self.yin['Nv'],
                             self.yin['Ev'], self.Vt)
        dp_dphip = cc.dp_dphip(psi, phi_p, self.yin['Nv'],
                               self.yin['Ev'], self.Vt)

        # Bernoulli function for current density calculation (m+1)
        B_plus = flux.bernoulli(+(psi[1:]-psi[:-1])/self.Vt)
        B_minus = flux.bernoulli(-(psi[1:]-psi[:-1])/self.Vt)
        Bdot_plus = flux.bernoulli_dot(+(psi[1:]-psi[:-1])/self.Vt)
        Bdot_minus = flux.bernoulli_dot(-(psi[1:]-psi[:-1])/self.Vt)

        # calculating current densities and their derivatives
        if discr == 'SG':  # Scharfetter-Gummel discretization
            jn, djn_dpsi1, djn_dpsi2, djn_dphin1, djn_dphin2 = \
                self._jn_SG(B_plus, B_minus, Bdot_plus, Bdot_minus, h)
            jp, djp_dpsi1, djp_dpsi2, djp_dphip1, djp_dphip2 = \
                self._jp_SG(B_plus, B_minus, Bdot_plus, Bdot_minus, h)

        elif discr == 'mSG':  # modified SG discretization
            jn, djn_dpsi1, djn_dpsi2, djn_dphin1, djn_dphin2 = \
                self._jn_mSG(B_plus, B_minus, Bdot_plus, Bdot_minus, h)
            jp, djp_dpsi1, djp_dpsi2, djp_dphip1, djp_dphip2 = \
                self._jp_mSG(B_plus, B_minus, Bdot_plus, Bdot_minus, h)

        else:
            raise Exception('Error: unknown current density '
                            + 'discretization scheme %s.' % discr)

        # store calculated current density
        self.sol['J'] = (jn + jp).mean()

        # spontaneous recombination rates (m)
        n0 = self.yin['n0'][1:-1]
        p0 = self.yin['p0'][1:-1]
        tau_n = self.yin['tau_n'][1:-1]
        tau_p = self.yin['tau_p'][1:-1]
        B_rad = self.yin['B'][1:-1]
        Cn = self.yin['Cn'][1:-1]
        Cp = self.yin['Cp'][1:-1]
        R_srh = rec.srh_R(n[1:-1], p[1:-1], n0, p0, tau_n, tau_p)
        R_rad = rec.rad_R(n[1:-1], p[1:-1], n0, p0, B_rad)
        R_aug = rec.auger_R(n[1:-1], p[1:-1], n0, p0, Cn, Cp)
        R = (R_srh + R_rad + R_aug)

        # recombination rates' derivatives
        dRsrh_dpsi = rec.srh_Rdot(n[1:-1], dn_dpsi[1:-1],
                                  p[1:-1], dp_dpsi[1:-1],
                                  n0, p0, tau_n, tau_p)
        dRrad_dpsi = rec.rad_Rdot(n[1:-1], dn_dpsi[1:-1],
                                  p[1:-1], dp_dpsi[1:-1],
                                  n0, p0, B_rad)
        dRaug_dpsi = rec.auger_Rdot(n[1:-1], dn_dpsi[1:-1],
                                    p[1:-1], dp_dpsi[1:-1],
                                    n0, p0, Cn, Cp)
        dR_dpsi = dRsrh_dpsi + dRrad_dpsi + dRaug_dpsi
        dRsrh_dphin = rec.srh_Rdot(n[1:-1], dn_dphin[1:-1], p[1:-1], 0,
                                   n0, p0, tau_n, tau_p)
        dRrad_dphin = rec.rad_Rdot(n[1:-1], dn_dphin[1:-1], p[1:-1], 0,
                                   n0, p0, B_rad)
        dRaug_dphin = rec.auger_Rdot(n[1:-1], dn_dphin[1:-1], p[1:-1], 0,
                                     n0, p0, Cn, Cp)
        dR_dphin = dRsrh_dphin + dRrad_dphin + dRaug_dphin
        dRsrh_dphip = rec.srh_Rdot(n[1:-1], 0, p[1:-1], dp_dphip[1:-1],
                                   n0, p0, tau_n, tau_p)
        dRrad_dphip = rec.rad_Rdot(n[1:-1], 0, p[1:-1], dp_dphip[1:-1],
                                   n0, p0, B_rad)
        dRaug_dphip = rec.auger_Rdot(n[1:-1], 0, p[1:-1], dp_dphip[1:-1],
                                     n0, p0, Cn, Cp)
        dR_dphip = dRsrh_dphip + dRrad_dphip + dRaug_dphip

        # stimulated emission
        ixa = self.ar_ix
        ixn = (n[ixa] < p[ixa])
        N = np.zeros_like(n[ixa])
        N[ixn] = n[ixa][ixn]
        N[~ixn] = p[ixa][~ixn]
        g0 = self.yin['g0'][ixa]
        N_tr = self.yin['N_tr'][ixa]
        gain = g0 * np.log(N/N_tr)
        ind_abs = np.where(gain < 0)
        gain[ind_abs] = 0.0  # no absoption

        # gain derivatives
        gain_dpsi = np.zeros_like(gain)
        gain_dpsi[ixn] = g0[ixn] * dn_dpsi[ixa][ixn] / n[ixa][ixn]
        gain_dpsi[~ixn] = g0[~ixn] * dp_dpsi[ixa][~ixn] / p[ixa][~ixn]
        gain_dphin = np.zeros_like(gain)
        gain_dphin[ixn] = g0[ixn] * dn_dphin[ixa][ixn] / n[ixa][ixn]
        gain_dphip = np.zeros_like(gain)
        gain_dphip[~ixn] = g0[~ixn] * dp_dphip[ixa][~ixn] / p[ixa][~ixn]
        for gdot in [gain_dpsi, gain_dphin, gain_dphip]:
            gdot[ind_abs] = 0  # ignore absoption

        # total gain and loss
        T = self.yin['wg_mode'][ixa]
        w_ar = w[ixa[1:-1]]
        alpha_fca = self._calculate_fca()
        alpha = self.alpha_i + self.alpha_m + alpha_fca
        total_gain = np.sum(gain * w_ar * T) - alpha

        # stimulated recombination rate
        S = self.sol['S']
        R_st = self.vg * gain * w_ar * T * S
        dRst_dS = self.vg * gain * w_ar * T
        dRst_dpsi = self.vg * gain_dpsi * w_ar * T * S
        dRst_dphin = self.vg * gain_dphin * w_ar * T * S
        dRst_dphip = self.vg * gain_dphip * w_ar * T * S

        # residual of the system (m*3 + 1)
        rvec = np.zeros(m*3 + 1)
        rvec[:m] = vrs.poisson_res(psi, n, p, h, w, self.yin['eps'],
                                   self.eps_0, self.q, self.yin['C_dop'])
        rvec[m:2*m] =  self.q*R*w - (jn[1:]-jn[:-1])
        rvec[m:2*m][ixa[1:-1]] += self.q*R_st
        rvec[2*m:3*m]  = -self.q*R*w - (jp[1:]-jp[:-1])
        rvec[2*m:3*m][ixa[1:-1]] -= self.q*R_st
        rvec[-1] = (self.beta_sp*np.sum(R_rad[ixa[1:-1]]*w_ar*T)
                    + self.vg*total_gain*S)

        # calculating Jacobian (m*3 + 1, m*3 + 1)
        # 1. Poisson's equation
        j11 = vrs.poisson_dF_dpsi(dn_dpsi, dp_dpsi, h, w, self.yin['eps'],
                                  self.eps_0, self.q)
        j12 = vrs.poisson_dF_dphin(dn_dphin, w, self.eps_0, self.q)
        j13 = vrs.poisson_dF_dphip(dp_dphip, w, self.eps_0, self.q)

        # 2. Electron current continuity equation
        j21 = vrs.jn_dF_dpsi(djn_dpsi1, djn_dpsi2, dR_dpsi, w, self.q, m)
        j22 = vrs.jn_dF_dphin(djn_dphin1, djn_dphin2, dR_dphin,
                              w, self.q, m)
        j23 = vrs.jn_dF_dphip(dR_dphip, w, self.q, m)
        j21[1, ixa[1:-1]] += self.q * dRst_dpsi
        j22[1, ixa[1:-1]] += self.q * dRst_dphin
        j23[ixa[1:-1]] += self.q * dRst_dphip
        j24 = np.zeros(m)
        j24[ixa[1:-1]] = self.q * dRst_dS

        # 3. Hole current continuity equation
        j31 = vrs.jp_dF_dpsi(djp_dpsi1, djp_dpsi2, dR_dpsi, w, self.q, m)
        j32 = vrs.jp_dF_dphin(dR_dphin, w, self.q, m)
        j33 = vrs.jp_dF_dphip(djp_dphip1, djp_dphip2, dR_dphip,
                              w, self.q, m)
        j31[1, ixa[1:-1]] -= self.q * dRst_dpsi
        j32[ixa[1:-1]] -= self.q * dRst_dphin
        j33[1, ixa[1:-1]] -= self.q * dRst_dphip
        j34 = np.zeros(m)
        j34[ixa[1:-1]] = -self.q * dRst_dS

        J = np.zeros((3*m+1, 3*m+1))
        # Poisson's equation
        J[np.arange(0, m), np.arange(2*m, 3*m)] = j13
        J[np.arange(0, m), np.arange(m, 2*m)] = j12
        J[np.arange(0, m-1), np.arange(1, m)] = j11[0, 1:]
        J[np.arange(0, m), np.arange(0, m)] = j11[1]
        J[np.arange(1, m), np.arange(0, m-1)] = j11[2, :-1]
        # electron current continuity equation
        J[np.arange(m, 2*m), np.arange(2*m, 3*m)] = j23
        J[np.arange(m, 2*m-1), np.arange(m+1, 2*m)] = j22[0, 1:]
        J[np.arange(m, 2*m), np.arange(m, 2*m)] = j22[1, :]
        J[np.arange(m+1, 2*m), np.arange(m, 2*m-1)] = j22[2, :-1]
        J[np.arange(m, 2*m-1), np.arange(1, m)] = j21[0, 1:]
        J[np.arange(m, 2*m), np.arange(0, m)] = j21[1, :]
        J[np.arange(m+1, 2*m), np.arange(0, m-1)] = j21[2, :-1]
        # hole current continuity equation
        J[np.arange(2*m, 3*m-1), np.arange(2*m+1, 3*m)] = j33[0, 1:]
        J[np.arange(2*m, 3*m), np.arange(2*m, 3*m)] = j33[1, :]
        J[np.arange(2*m+1, 3*m), np.arange(2*m, 3*m-1)] = j33[2, :-1]
        J[np.arange(2*m, 3*m), np.arange(m, 2*m)] = j32
        J[np.arange(2*m, 3*m-1), np.arange(1, m)] = j31[0, 1:]
        J[np.arange(2*m, 3*m), np.arange(0, m)] = j31[1, :]
        J[np.arange(2*m+1, 3*m), np.arange(0, m-1)] = j31[2, :-1]
        # rightmost column -- derivatives w.r.t. S
        J[m:2*m, -1] = j24
        J[2*m:3*m, -1] = j34
        # bottom row -- photon density rate equation
        J[-1, :m][ixa[1:-1]] = (self.beta_sp * dRrad_dpsi[ixa[1:-1]]
                                * w_ar * T
                                + self.vg * gain_dpsi * w_ar * T * S)
        J[-1, m:2*m][ixa[1:-1]] = (self.beta_sp * dRrad_dphin[ixa[1:-1]]
                                   * w_ar * T
                                   + self.vg * gain_dphin * w_ar * T * S)
        J[-1, 2*m:3*m][ixa[1:-1]] = (self.beta_sp * dR_dphip[ixa[1:-1]]
                                     * w_ar * T * S
                                     + self.vg * gain_dphip * w_ar * T * S)
        J[-1, -1] = self.vg * total_gain

        # rightmost column -- derivatives w.r.t. S
        J[m:2*m, -1] = j24
        J[2*m:3*m, -1] = j34
        # bottom row -- photon density RE
        inds = np.where(ixa[1:-1])[0]
        J[-1, inds] = (self.beta_sp * dRrad_dpsi[ixa[1:-1]]
                                * w_ar * T
                                + self.vg * gain_dpsi * w_ar * T * S)
        J[-1, inds+m] = (self.beta_sp * dRrad_dphin[ixa[1:-1]]
                                   * w_ar * T
                                   + self.vg * gain_dphin * w_ar * T * S)
        J[-1, inds+2*m] = (self.beta_sp * dR_dphip[ixa[1:-1]]
                                     * w_ar * T * S
                                     + self.vg * gain_dphip * w_ar * T * S)
        J[-1, -1] = self.vg * total_gain
        J = sparse.csc_matrix(J)

        return J, rvec

    def lasing_step(self, omega=0.1, omega_S=(1.0, 0.1), discr='mSG'):
        """
        Perform a single Newton step for the lasing problem.

        Parameters
        ----------
        omega : float
            Damping parameter for potentials.
        omega_S : (float, float)
            Dampling parameter for photon density `S`. First value is used
            for increasing `S`, second -- for decreasing `S`.
        discr : str
            Current density discretiztion scheme.

        """
        # residual vector and Jacobian for transport problem
        J, rvec = self._lasing_system(discr)

        # solve the system
        dx = sparse.linalg.spsolve(J, -rvec)
        x = np.hstack((self.sol['psi'][1:-1],
                       self.sol['phi_n'][1:-1],
                       self.sol['phi_p'][1:-1],
                       np.array([self.sol['S']])))
        fluct = newton.l2_norm(dx) / newton.l2_norm(x)
        self.fluct.append(fluct)
        self.iterations += 1
 
        # update solution
        m = self.npoints - 2
        self.sol['psi'][1:-1] += dx[:m]*omega
        self.sol['phi_n'][1:-1] += dx[m:2*m]*omega
        self.sol['phi_p'][1:-1] += dx[2*m:3*m]*omega
        self.sol['n'] = cc.n(self.sol['psi'], self.sol['phi_n'],
                             self.yin['Nc'], self.yin['Ec'], self.Vt)
        self.sol['p'] = cc.p(self.sol['psi'], self.sol['phi_p'],
                             self.yin['Nv'], self.yin['Ev'], self.Vt)
        if dx[-1] > 0:
            self.sol['S'] += dx[-1] * omega_S[0] 
        else:
            self.sol['S'] += dx[-1] * omega_S[1]

        return fluct


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sample_design import sd

    plt.rc('lines', linewidth=0.7)
    plt.rc('figure.subplot', left=0.15, right=0.85)

    print('Creating an instance of LaserDiode1D...', end=' ')
    ld = LaserDiode1D(design=sd, ar_inds=3,
                      L=3000e-4, w=100e-4,
                      R1=0.95, R2=0.05,
                      lam=0.87e-4, ng=3.9,
                      alpha_i=0.5, beta_sp=1e-4)
    print('Complete.')

    # 1. nonuniform mesh
    print('Generating a nonuniform mesh...', end=' ')
    ld.gen_nonuniform_mesh(param='Eg', y_ext=[0.3, 0.3])
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
    nsteps = 500
    ld.make_dimensionless()
    print('Solving drift-diffution system at small forward bias...',
          end=' ')
    ld.transport_init(0.1)
    for _ in range(nsteps):
        ld.lasing_step(0.1, [1.0, 0.1], 'mSG')
    print('Complete.')
    ld.original_units()
    plt.figure('Small forward bias')
    plt.plot(x, ld.yin['Ec']-ld.sol['psi'], color='b')
    plt.plot(x, ld.yin['Ev']-ld.sol['psi'], color='b')
    plt.plot(x, -ld.sol['phi_n'], color='g', ls=':')
    plt.plot(x, -ld.sol['phi_p'], color='r', ls=':')
    plt.xlabel(r'$x$ ($\mu$m)')
    plt.ylabel('Energy (eV)')

    plt.figure('Convergence')
    plt.plot(ld.fluct)
    plt.xlabel('Iteration number')
    plt.ylabel('Fluctuation')
    plt.yscale('log')
