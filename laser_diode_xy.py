# -*- coding: utf-8 -*-
"""
2D (vertical-lateral or x-y) laser diode model.
"""

import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
import design
import constants as const
import waveguide as wg
import units


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
        assert p in self.params_n
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
        n_eff, modes = wg.solve_wg_2d(x, y, n, self.lam, n_modes, nx, ny)
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
        self.wg_mode = self.wg_fun(x=x, y=y, grid=False)


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
    x, y, Eg = ld.get_value('Eg')

    plt.close('all')
    plt.figure()
    plt.scatter(y, x, c=Eg, marker='s', s=2)
    plt.show()
