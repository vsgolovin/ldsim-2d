# -*- coding: utf-8 -*-
"""
2D (vertical-lateral or x-y) laser diode model.
"""

import warnings
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy import sparse
import design
import constants as const
import waveguide as wg
import units
import carrier_concentrations as cc
import equilibrium as eq
import newton


class LaserDiode(object):

    params_n = ['Ev', 'Ec', 'Eg', 'Nd', 'Na', 'C_dop', 'Nc', 'Nv', 'mu_n',
                'mu_p', 'tau_n', 'tau_p', 'B', 'Cn', 'Cp', 'eps', 'n_refr',
                'fca_e', 'fca_h']
    params_active = ['g0', 'N_tr']
    params_b = ['Ev', 'Ec', 'Nc', 'Nv', 'mu_n', 'mu_p']

    def __init__(self, dsgn, L, R1, R2, lam, ng, alpha_i, beta_sp):
        """
        Class for storing all the model parameters of a 2D laser diode.

        Parameters
        ----------
        dsgn : design.Design2D
            Vertical-lateral laser design.
        L : float
            Resonator length (cm).
        w : float
            Stripe width (cm).
        R1 : float
            Back (x=0) mirror reflectivity (0<`R1`<=1).
        R2 : float
            Front (x=L) mirror reflectivity (0<`R2`<=1).
        lam : float
            Operating wavelength (cm).
        ng : number
            Group refrative index.
        alpha_i : number
            Internal optical loss (cm-1). Should not include free-carrier
            absorption.
        beta_sp : float
            Spontaneous emission factor, i.e., the fraction of spontaneous
            emission that is coupled with the lasing mode.

        """
        # check if all the necessary parameters were specified
        # and if there is an active region
        assert isinstance(dsgn, design.Design2D)
        has_active_region = False
        self.ar_inds = list()
        for i, layer in enumerate(dsgn.epi):
            assert True not in [list(yi) == [np.nan] for yi in layer.d.values()]
            if layer.active:
                has_active_region = True
                self.ar_inds.append(i)
        assert has_active_region
        self.dsgn = dsgn

        # constants
        self.Vt = const.kb * const.T
        self.q = const.q
        self.eps_0 = const.eps_0

        # device parameters
        self.L = L
        self.R1 = R1
        self.R2 = R2
        assert all([r > 0 and r < 1 for r in (R1, R2)])
        self.alpha_m = 1/(2*L) * np.log(1/(R1*R2))
        self.lam = lam
        self.photon_energy = const.h * const.c / lam
        self.ng = ng
        self.vg = const.c / ng
        self.n_eff = None
        self.gamma = None
        self.psi_bi = np.zeros(0)
        self.alpha_i = alpha_i
        self.beta_sp = beta_sp

        self.mesh = dict()
        self.vxn = dict()  # parameters' values at x mesh nodes
        self.vxb = dict()  # -/- volume boundaries
        self.wg_fun = lambda x, y: 0  # waveguide mode function
        self.wg_mode = np.array([])   # array for current grid
        self.sol = dict()
        self.is_dimensionless = False

    def gen_uniform_mesh(self, nx, ny):
        """
        Generate a uniform 2D mesh with `nx` nodes along the x axis and
        `ny` nodes along the y axis.
        """
        xb = np.linspace(0, self.dsgn.get_thickness(), nx + 1)
        yb = np.linspace(0, self.dsgn.get_width(), ny + 1)
        self.mesh = self._make_mesh(xb, yb)

    def gen_nonuniform_mesh(self, step_min=1e-7, step_max=20e-7,
                            step_uni=5e-8, sigma=100e-7,
                            y_ext=[0., 0.], ny=100):
        """
        Generate mesh which is uniform along the y axis and nonuniform
        along the x axis. Uses local bandgap change to decide x mesh step.
        """
        def gauss(x, mu, sigma):
            return np.exp(-(x-mu)**2 / (2*sigma**2))

        # uniform y mesh
        yb = np.linspace(0, self.dsgn.get_width(), ny)

        # temporary uniform x mesh
        thickness = self.dsgn.get_thickness()
        nx = int(round(thickness / step_uni))
        x = np.linspace(0, thickness, nx)

        # calculate bandgap
        Eg = np.zeros(len(x) + 2)
        Eg[1:-1] = self.dsgn.epi.calculate('Eg', x)
        # external values for fine grid at boundaries
        for i, j in zip(range(2), [1, -2]):
            if not isinstance(y_ext[i], (float, int)):
                y_ext[i] = Eg[j]
        Eg[0] = y_ext[0]
        Eg[-1] = y_ext[1]

        # function for choosing local step size
        f = np.abs(Eg[2:] - Eg[:-2])  # change of y at every point
        fg = np.zeros_like(f)  # convolution for smoothing
        for i, xi in enumerate(x):
            g = gauss(x, xi, sigma)
            fg[i] = np.sum(f * g)
        fg_fun = interp1d(x, fg / fg.max())

        # generate new mesh
        k = step_max - step_min
        new_mesh = list()
        xi = 0
        while xi <= thickness:
            new_mesh.append(xi)
            xi += step_min + k*(1 - fg_fun(xi))
        xb = np.array(new_mesh)
        self.mesh = self._make_mesh(xb, yb)

    def _make_mesh(self, xb, yb):
        "Generate mesh from given volume boundaries."
        msh = dict()
        msh['xb'] = xb
        msh['xn'] = (xb[1:] + xb[:-1]) / 2          # nodes
        msh['hx'] = msh['xn'][1:] - msh['xn'][:-1]  # spacing between nodes
        msh['wx'] = xb[1:] - xb[:-1]                # finite volume sizes
        msh['yb'] = yb
        msh['yn'] = (yb[1:] + yb[:-1]) / 2
        msh['hy'] = msh['yn'][1:] - msh['yn'][:-1]
        msh['wy'] = yb[1:] - yb[:-1]
        msh['ixa'] = self.dsgn.epi._get_ixa(msh['xn'])

        # number of x mesh points for every yn
        my = len(msh['yn'])
        msh['mx'] = np.zeros(my, dtype=int)
        msh['tc'] = np.zeros(my, dtype=bool)
        msh['bc'] = np.zeros(my, dtype=bool)
        for j, yj in enumerate(msh['yn']):
            for xi in msh['xn']:
                if self.dsgn.inside(xi, yj):
                    msh['mx'][j] += 1
                msh['tc'][j] = self.dsgn.inside_top_contact(yj)
                msh['bc'][j] = self.dsgn.inside_bottom_contact(yj)

        # mask for selecting contact nodes
        msh['ixc'] = np.full(np.sum(msh['mx']), False)
        k = 0
        for j, mx in enumerate(msh['mx']):
            if msh['bc'][j]:
                msh['ixc'][k] = True
            if msh['tc'][j]:
                msh['ixc'][k+mx-1] = True
            k += mx

        return msh

    def get_flattened_2d_mesh(self):
        x = np.zeros(np.sum(self.mesh['mx']))
        y = np.zeros_like(x)
        i = 0
        for j, mx in enumerate(self.mesh['mx']):
            x[i:i+mx] = self.mesh['xn'][:mx]
            y[i:i+mx] = self.mesh['yn'][j]
            i += mx
        return x, y

    def get_value(self, p):
        "Get mesh nodes and `p` parameter values as 1D arrays."
        x, y = self.get_flattened_2d_mesh()
        v = np.zeros_like(x)
        i = 0
        for mx in self.mesh['mx']:
            v[i:i+mx] = self.vxn[p][:mx]
            i += mx
        return x, y, v

    def solve_waveguide(self, nx=1000, ny=100, n_modes=3,
                        remove_layers=(0, 0)):
        """
        Calculate 2D laser mode profile. Finds `n_modes` solutions of the
        eigenvalue problem with the highest eigenvalues (effective
        indices) and pick the one with the highest optical confinement
        factor (active region overlap).

        Parameters
        ----------
        nx : int, optional
            Number of uniform grid nodes along the x (vertical) axis.
        ny : int, optional
            Number of uniform grid nodes along the x (vertical) axis.
        n_modes: int, optional
            Number of calculated eigenproblem solutions.
        remove_layers : (int, int), optional
            Number of layers to exclude from calculated refractive index
            profile at each side of the device. Useful to exclude contact
            layers.

        """
        # x and y coordinates for uniform mesh volume boundaries
        xb = np.linspace(0, self.dsgn.get_thickness(), nx + 1)
        yb = np.linspace(0, self.dsgn.get_width(), ny + 1)

        # remove from xb and yb points belonging to layers
        # that are to be removed from calculation
        inds, _ = self.dsgn.epi._inds_dx(xb)
        i2r = list()  # layer indices to remove
        i2r += [i for i in range(remove_layers[0])]
        imax = len(self.dsgn.epi)
        i2r += [i for i in range(imax - remove_layers[1], imax)]
        ix = np.ones_like(inds, dtype=bool)  # array of True
        for i in i2r:
            ix &= (inds != i)
        xb = xb[ix]
        nx = len(xb) - 1
        # create rectangular uniform mesh
        msh = self._make_mesh(xb, yb)

        # create 1D arrays of all 2D mesh nodes
        x = np.tile(msh['xn'], ny)
        y = np.repeat(msh['yn'], nx)

        # calculate refractive for each node
        n_1D = self.dsgn.epi.calculate('n_refr', msh['xn'])
        n = np.ones(len(msh['xn']) * ny)
        for i, mx in enumerate(msh['mx']):
            n[i*nx:i*nx+mx] = n_1D[:mx]

        # solve the waveguide problem
        # and choose the mode with the highest confinement factor
        lam = self.lam
        if self.is_dimensionless:
            lam *= units.x
        n_eff, modes = wg.solve_wg_2d(x, y, n, lam, n_modes, nx, ny)
        w = msh['wx'][0] * msh['wy'][0]  # area of every finite volume
        ixa = np.tile(msh['ixa'], ny)
        gammas = np.zeros(n_modes)
        for i in range(n_modes):
            gammas[i] = np.sum(modes[ixa, i]) * w
        i = np.argmax(gammas)
        self.gamma = gammas[i]
        self.n_eff = n_eff[i]
        mode = modes[:, i]

        # interpolate mode and store corresponding function
        self.wg_fun = RectBivariateSpline(
            msh['xn']/units.x, msh['yn']/units.x,
            mode.reshape((ny, nx)).T * units.x**2)

        return msh['xn'], msh['yn'], mode.reshape(ny, -1)

    def calc_all_params(self):
        "Calculate all parameters' values at mesh nodes and boundaries."
        epi = self.dsgn.epi
        inds, dx = epi._inds_dx(self.mesh['xn'])  # nodes
        for p in self.params_n:
            self.vxn[p] = epi.calculate(p, self.mesh['xn'], inds, dx)
        ixa = self.mesh['ixa']
        for p in self.params_active:
            self.vxn[p] = epi.calculate(p, self.mesh['xn'][ixa],
                                        inds=inds[ixa], dx=dx[ixa])
        inds, dx = epi._inds_dx(self.mesh['xb'])  # boundaries
        for p in self.params_b:
            self.vxb[p] = epi.calculate(p, self.mesh['xb'], inds, dx)

        # flattened 2D array for waveguide mode profile
        x, y = self.get_flattened_2d_mesh()
        if not self.is_dimensionless:
            x /= units.x
            y /= units.x
        if self.n_eff is not None:
            self.wg_mode = self.wg_fun(x=x, y=y, grid=False)

    def make_dimensionless(self):
        "Make every parameter dimensionless."
        if self.is_dimensionless:
            return

        # constants
        self.Vt /= units.V
        self.q /= units.q
        self.eps_0 /= units.q / (units.x*units.V)

        # device parameters
        self.L /= units.x
        self.alpha_m /= 1/units.x
        self.lam /= units.x
        self.photon_energy /= units.E
        self.vg /= units.x/units.t
        self.alpha_i /= 1/units.x

        # arrays
        for key in ['xn', 'xb', 'hx', 'wx',
                    'yn', 'yb', 'hy', 'wy']:
            self.mesh[key] /= units.x
        for d in [self.vxn, self.vxb, self.sol]:
            for key in d:
                d[key] /= units.dct[key]
        self.wg_mode /= 1/units.x**2
        self.psi_bi /= units.V

        self.is_dimensionless = True

    def original_units(self):
        "Convert every parameter back to original units."
        if not self.is_dimensionless:
            return

        # constants
        self.Vt *= units.V
        self.q *= units.q
        self.eps_0 *= units.q / (units.x*units.V)

        # device parameters
        self.L *= units.x
        self.alpha_m *= 1/units.x
        self.lam *= units.x
        self.photon_energy *= units.E
        self.vg *= units.x/units.t
        self.alpha_i *= 1/units.x

        # arrays
        for key in ['xn', 'xb', 'hx', 'wx',
                    'yn', 'yb', 'hy', 'wy']:
            self.mesh[key] *= units.x
        for d in [self.vxn, self.vxb, self.sol]:
            for key in d:
                d[key] *= units.dct[key]
        self.wg_mode *= 1/units.x**2
        self.psi_bi *= units.V

        self.is_dimensionless = False

    # local charge neutrality
    def make_lcn_solver(self):
        """
        Make a `NewtonSolver` for electrostatic potential distribution
        along the x axis at equilibrium assuming local charge neutrality.
        """
        v = self.vxn  # values at x mesh nodes

        def f(psi):
            n = cc.n(psi, 0, v['Nc'], v['Ec'], self.Vt)
            p = cc.p(psi, 0, v['Nv'], self.vxn['Ev'], self.Vt)
            ndot = cc.dn_dpsi(psi, 0, v['Nc'], self.vxn['Ec'], self.Vt)
            pdot = cc.dp_dpsi(psi, 0, v['Nv'], self.vxn['Ev'], self.Vt)
            return (v['C_dop'] - n + p, -ndot + pdot)

        # initial guess using Boltzmann statistics
        ni = eq.intrinsic_concentration(v['Nc'], v['Nv'],
                                        v['Ec'], v['Ev'], self.Vt)
        Ei = eq.intrinsic_level(v['Nc'], v['Nv'],
                                v['Ec'], v['Ev'], self.Vt)
        Ef_i = eq.Ef_lcn_boltzmann(v['C_dop'], ni, Ei, self.Vt)

        return newton.NewtonSolver(f, Ef_i, lambda A, b: b/A)

    def solve_lcn(self, maxiter=100, fluct=1e-12, omega=1.0):
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
        sol = self.make_lcn_solver()
        sol.solve(maxiter, fluct, omega)
        if sol.fluct[-1] > fluct:
            warnings.warn('LaserDiode.solve_lcn(): fluctuation ' +
                          '%e exceeds %e.' % (sol.fluct[-1], fluct))

        self.vxn['psi_lcn'] = sol.x.copy()
        self.vxn['n0'] = cc.n(psi=self.vxn['psi_lcn'], phi_n=0,
                              Nc=self.vxn['Nc'], Ec=self.vxn['Ec'],
                              Vt=self.Vt)
        self.vxn['p0'] = cc.p(psi=self.vxn['psi_lcn'], phi_p=0,
                              Nv=self.vxn['Nv'], Ev=self.vxn['Ev'],
                              Vt=self.Vt)

    def make_equilibrium_solver(self):
        """
        Make a `NewtonSolver` for electrostatic potential distribution
        in the x-y plane at equilibrium (zero external bias).
        """
        assert 'psi_lcn' in self.vxn
        _, _, psi_0 = self.get_value('psi_lcn')
        pts = len(psi_0) - ld.mesh['ixc'].sum()
        v = self.vxn
        msh = self.mesh
        hy = msh['hy']
        my = len(msh['yn'])

        def f(psi):
            rvec = np.zeros(pts)
            J = sparse.lil_matrix((pts, pts))

            # iterate over vertical slices
            i1 = 0  # indexing psi (with contact nodes)
            i2 = 0  # indexing rvec and J
            for j in range(my):

                # number of nodes in current, previous and next slices
                mx = msh['mx'][j]
                mxb, mxf = 0, 0  # back / forward
                if j > 0:
                    mxb = msh['mx'][j-1]
                if j < (my - 1):
                    mxf = msh['mx'][j+1]

                # some aliases
                psi_j = psi[i1:i1+mx]
                eps = v['eps'][:mx]
                hx = msh['hx'][:mx-1]
                wx = msh['wx'][:mx]
                wy = msh['wy'][j]

                # carrier densities and their derivatives w.r.t. potential
                n = cc.n(psi_j, 0, v['Nc'][:mx], v['Ec'][:mx], self.Vt)
                ndot = cc.dn_dpsi(psi_j, 0, v['Nc'][:mx], v['Ec'][:mx],
                                  self.Vt)
                p = cc.p(psi_j, 0, v['Nv'][:mx], v['Ev'][:mx], self.Vt)
                pdot = cc.dp_dpsi(psi_j, 0, v['Nv'][:mx], v['Ev'][:mx],
                                  self.Vt)

                # residuals (without dpsi/dy terms)
                rj = self.q/self.eps_0 * (v['C_dop'][:mx] - n + p) * wx*wy
                rj[:-1] += eps[:-1] * (psi_j[1:] - psi_j[:-1]) / hx * wy
                rj[1:] += -eps[1:] * (psi_j[1:] - psi_j[:-1]) / hx * wy

                # Jacobian, 3 diagonals: (1, 0, -1) -- top, main, bottom
                # y derivatives are considered later
                md = self.q / self.eps_0 * (pdot - ndot) * wx * wy
                md[:-1] += -eps[:-1] / hx * wy
                md[1:] += -eps[1:] / hx * wy
                td = eps[:-1] / hx * wy
                bd = eps[1:] / hx * wy

                # check if top or bottom nodes are contacts
                mx2 = mx
                if msh['tc'][j]:
                    mx2 -= 1
                if msh['bc'][j]:
                    mx2 -= 1
                    k = 1
                else:
                    k = 0

                # forward derivative w.r.t. y
                if mxf > 0:
                    m = min(mxf, mx)
                    psi_f = psi[i1+mx : i1+mx+m]
                    rj[:m] += eps[:m] * (psi_f - psi_j[:m]) / hy[j] * wx[:m]
                    td2 = eps[:m] / hy[j] * wx[:m]
                    md[:m] += -eps[:m] / hy[j] * wx[:m]

                    if msh['bc'][j]:
                        k1, k2 = 0, 1
                        m -= 1
                    elif msh['bc'][j+1]:
                        k1, k2 = 1, 1
                        m -= 1
                    else:
                        k1, k2 = 0, 0
                    if ((mx <= mxf and msh['tc'][j])
                       or (mx >= mxf and msh['tc'][j+1])):
                        m -= 1
                    inds = np.arange(i2 + k1, i2 + k1 + m)
                    J[inds, inds+mx2] = td2[k2:k2+m]

                # same for back derivative
                if mxb > 0:
                    m = min(mxb, mx)
                    psi_b = psi[i1-mxb : i1-mxb+m]
                    rj[:m] += -eps[:m] * (psi_j[:m] - psi_b) / hy[j-1] * wx[:m]
                    bd2 = -eps[:m] / hy[j-1] * wx[:m]
                    md[:m] += eps[:m] / hy[j-1] * wx[:m]

                    if msh['bc'][j]:
                        k1, k2 = 0, 1
                        m -= 1
                    elif msh['bc'][j-1]:
                        k1, k2 = 1, 1
                        m -= 1
                    else:
                        k1, k2 = 0, 0
                    if ((mx <= mxb and msh['tc'][j])
                       or (mx >= mxb and msh['tc'][j-1])):
                        m -= 1
                    inds = np.arange(i2 + k1, i2 + k1 + m)
                    mxb2 = mxb - int(msh['tc'][j-1]) - int(msh['bc'][j-1])
                    J[inds, inds-mxb2] = bd2[k2:k2+m]

                # fill rvec and J (3 main diagonals) with calculated values
                rvec[i2:i2+mx2] = rj[k:k+mx2]
                inds = np.arange(i2, i2 + mx2)
                J[inds, inds] = md[k:k+mx2]
                J[inds[:-1], inds[1:]] = td[k:k+mx2-1]
                J[inds[1:], inds[:-1]] = bd[k:k+mx2-1]
                i1 += mx
                i2 += mx2

            return rvec, J.tocsc()

        return newton.NewtonSolver(f, psi_0,
                                   sparse.linalg.spsolve,
                                   ~msh['ixc'])

    def solve_equilibrium(self, maxiter=100, fluct=1e-15, omega=1.0):
        """
        Calculate electrostatic potential distribution in the x-y plane at
        equilibrium (zero external bias). Uses Newtons's method implemented
        in `NewtonSolver`.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of Newtons's method iterations.
        fluct : float, optional
            Fluctuation of solution that is needed to stop iterating before
            reaching `maxiter` steps.
        omega : float, optional
            Damping parameter.

        """
        sol = self.make_equilibrium_solver()
        sol.solve(maxiter, fluct, omega)
        if sol.fluct[-1] > fluct:
            warnings.warn('LaserDiode.solve_equilibrium(): fluctuation ' +
                          ('%e exceeds %e.' % (sol.fluct[-1], fluct)))
        self.psi_bi = sol.x.copy()
        self.sol['psi'] = sol.x.copy()
        self.sol['phi_n'] = np.zeros_like(sol.x)
        self.sol['phi_p'] = np.zeros_like(sol.x)
        self._update_densities()
        return sol

    def _update_densities(self):
        """
        Update electron and hole densities using currently stored potentials.
        """
        Nc = self.get_value('Nc')[2]
        Ec = self.get_value('Ec')[2]
        Nv = self.get_value('Nv')[2]
        Ev = self.get_value('Ev')[2]
        psi = self.sol['psi']
        phi_n = self.sol['phi_n']
        phi_p = self.sol['phi_p']
        self.sol['n'] = cc.n(psi, phi_n, Nc, Ec, self.Vt)
        self.sol['dn_dpsi'] = cc.dn_dpsi(psi, phi_n, Nc, Ec, self.Vt)
        self.sol['dn_dphin'] = cc.dn_dphin(psi, phi_n, Nc, Ec, self.Vt)
        self.sol['p'] = cc.p(psi, phi_p, Nv, Ev, self.Vt)
        self.sol['dp_dpsi'] = cc.dp_dpsi(psi, phi_p, Nv, Ev, self.Vt)
        self.sol['dp_dphip'] = cc.dp_dphip(psi, phi_p, Nv, Ev, self.Vt)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sample_design import epi

    dsgn = design.Design2D(epi, 20e-4)
    dsgn.add_trenches(2e-4, 4e-4, 2.0e-4)
    dsgn.set_top_contact(10e-4)
    dsgn.set_bottom_contact(18e-4)

    ld = LaserDiode(dsgn, 2000e-4, 0.3, 0.3, 0.87e-4, 3.9, 0.5, 1e-4)
    ld.solve_waveguide(nx=1000, ny=100, n_modes=3,
                       remove_layers=(1, 1))
    ld.gen_nonuniform_mesh()
    ld.calc_all_params()
    ld.make_dimensionless()
    ld.solve_lcn()
    ld.solve_equilibrium()
    ld.original_units()
    x, y = ld.get_flattened_2d_mesh()

    plt.close('all')
    plt.figure('psi')
    plt.scatter(y, x, c=ld.psi_bi, marker='s')
    plt.figure('n')
    plt.scatter(y, x, c=ld.sol['n'], marker='s')
    plt.figure('p')
    plt.scatter(y, x, c=ld.sol['p'], marker='s')
    plt.show()
