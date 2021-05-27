# -*- coding: utf-8 -*-
"""
2-dimensional model of a diode laser.
"""

import numpy as np
from scipy import sparse, optimize, interpolate
from ld_1d import LaserDiode1D
import units
from newton import l2_norm
import recombination as rec


class LaserDiode2D(LaserDiode1D):
    def __init__(self, design, ar_inds, L, w, R1, R2,
                 lam, ng, alpha_i, beta_sp):
        LaserDiode1D.__init__(self, design, ar_inds, L, w, R1, R2,
                              lam, ng, alpha_i, beta_sp)
        self.nz = 1             # number of z grid nodes
        self.dz = L             # longitudinal grid step
        self.zin = np.zeros(1)  # z grid nodes
        self.zbn = np.zeros(1)  # z grid volume boundaries
        self.sol2d = list()     # solution at every slice
        self.ndim = 1           # initialize as 1D

        # photon distribution along z
        self.Sf = np.zeros(self.nz)  # forward-propagating
        self.Sb = np.zeros(self.nz)  # backward-propagating
        self.G = np.zeros(self.nz)   # net gain (Gamma*g - alpha_i)

    def make_dimensionless(self):
        LaserDiode1D.make_dimensionless(self)
        self.dz /= units.x
        self.zin /= units.x
        self.zbn /= units.x
        self.Sf /= units.n * units.x
        self.Sb /= units.n * units.x
        self.G /= (1 / units.x)

    def original_units(self):
        LaserDiode1D.original_units(self)
        self.dz *= units.x
        self.zin *= units.x
        self.zbn *= units.x
        self.Sf *= units.n * units.x
        self.Sb *= units.n * units.x
        self.G *= (1 / units.x)

    def to_2D(self, n):
        self.nz = n
        self.sol2d = [dict() for _ in range(n)]
        for i in range(n):
            for key in ('psi', 'phi_n', 'phi_p', 'n', 'p'):
                self.sol2d[i][key] = self.sol[key].copy()
                self.sol2d[i]['S'] = self.sol['S']
        self.ndim = 2
        self.dz = self.L / n
        self.zin = np.arange(self.dz / 2, self.L, self.dz)
        self.zbn = np.linspace(0, self.L, self.nz + 1)
        self._calculate_G()  # fills self.G

        self._fit_Sf_Sb()
        S = self._calculate_S()
        for i in range(self.nz):
            self.sol2d[i]['S'] = S[i]

    def _fit_Sf_Sb(self, div=3):
        assert self.ndim == 2
        S = np.array([d['S'] for d in self.sol2d])

        def sumsq(S0):
            Sf, Sb = self._calculate_Sf_Sb(S0)
            S_new = self._calculate_S(Sf, Sb)
            return np.sum((S_new-S)**2)

        res = optimize.minimize(sumsq, S[0]/div)
        self.Sf, self.Sb = self._calculate_Sf_Sb(res.x)

    def _calculate_G(self):
        "Calculate net gain for every slice and store in `self.G`."
        assert self.ndim == 2
        # material parameters are same for all slices
        ixa = self.ar_ix
        g0 = self.yin['g0'][ixa]
        N_tr = self.yin['N_tr'][ixa]
        w = (self.xbn[1:] - self.xbn[:-1])[ixa[1:-1]]
        T = self.yin['wg_mode'][ixa]

        # iterate over slices
        self.G = np.zeros(self.nz)
        for i in range(self.nz):
            n = self.sol2d[i]['n'][ixa]
            p = self.sol2d[i]['p'][ixa]
            N = p.copy()
            ixn = n < p
            N[ixn] = n[ixn]
            g = g0 * np.log(N / N_tr)
            g[g < 0] = 0.0
            alpha_fca = self._calculate_fca(self.sol2d[i]['n'],
                                            self.sol2d[i]['p'])
            self.G[i] = np.sum(g * T * w) - (self.alpha_i + alpha_fca)

    def _calculate_Sf_Sb(self, Sf0):
        """
        Returns forward- and backward-propagating forward densities `Sf`
        and `Sb` assuming `Sf[0] = Sf0`.
        Ignores spontaneous emission.
        """
        assert self.ndim == 2

        # radiative recombination
        ixa = self.ar_ix
        T = self.yin['wg_mode'][ixa]
        w = (self.xbn[1:] - self.xbn[:-1])[ixa[1:-1]]
        n0 = self.yin['n0'][ixa]
        p0 = self.yin['p0'][ixa]
        B = self.yin['B'][ixa]
        gamma = self.beta_sp / (2 * self.vg) * self.dz
        dS_rad = np.zeros(self.nz)
        for i in range(self.nz):
            n = self.sol2d[i]['n'][ixa]
            p = self.sol2d[i]['p'][ixa]
            R_rad = rec.rad_R(n, p, n0, p0, B)
            dS_rad[i] = gamma * np.sum(R_rad * T * w)

        # forward propagation
        # Gdz = np.cumsum(self.G * self.dz)
        Sf = np.zeros(self.nz + 1)
        Sf[0] = Sf0
        for i in range(self.nz):
            Sf[i+1] = Sf[i] * np.exp(self.G[i] * self.dz) + dS_rad[i]

        # Sb = np.zeros(self.nz + 1)
        # Sb[0] = Sf0 / self.R1
        # for i in range(self.nz):
        #     Sb[i+1] = Sb[i] / np.exp(self.G[i] * self.dz)

        # back propagation
        # Gdz = np.cumsum(self.G[::-1] * self.dz)[::-1]
        Sb = np.zeros(self.nz + 1)
        Sb[-1] = Sf[-1] * self.R2
        for i in range(self.nz, 0, -1):
            Sb[i-1] = Sb[i] * np.exp(self.G[i-1]*self.dz) + dS_rad[i-1]

        return Sf, Sb

    def _transport_system_2D(self, Phi, discr):
        J, r = self._transport_system(discr, laser=True, save_J=False,
                                      save_Isp=False)
        r[-1] += self.vg*(Phi + self.sol['S'] * self.alpha_m)
        J[-1, -1] += self.vg * self.sol['S'] * self.alpha_m
        return J, r

    def lasing_step_2D(self, discr='mSG', niter=10, omega=0.1,
                       omega_S=(1.0, 0.1)):
        # Sf = self.Sf
        # Sb = self.Sb
        nz = len(self.sol2d)
        Sf_fun = interpolate.InterpolatedUnivariateSpline(self.zbn,
                                                          self.Sf,
                                                          k=3)
        Sfdot_fun = Sf_fun.derivative()
        Sb_fun = interpolate.InterpolatedUnivariateSpline(self.zbn,
                                                          self.Sb,
                                                          k=3)
        Sbdot_fun = Sb_fun.derivative()
        Phi = Sbdot_fun(self.zin) - Sfdot_fun(self.zin)
        # Phi = (Sb[1:] - Sb[:-1] - Sf[1:] + Sf[:-1]) / self.dz
        # Phi = np.zeros(nz)
        # Phi[1:-1] = (Sb[2:] - Sb[1:-1] - Sf[1:-1] + Sf[:-2]) / self.dz
        # Phi[0] = (Sb[1] - Sb[0] - Sf[0] + Sb[0] * self.R1) / self.dz
        # Phi[-1] = (Sf[-1] * self.R2 - Sb[-1] - Sf[-1] + Sf[-2]) / self.dz
        m = self.npoints - 2

        fluct = 0
        for i in range(nz):
            self.sol = self.sol2d[i]
            for _ in range(niter):
                J, r = self._transport_system_2D(Phi[i], discr)
                dx = sparse.linalg.spsolve(J, -r)
                self.sol['psi'][1:-1] += dx[:m] * omega
                self.sol['phi_n'][1:-1] += dx[m:2*m] * omega
                self.sol['phi_p'][1:-1] += dx[2*m:3*m] * omega
                dS = dx[-1]
                if dS > 0:
                    self.sol['S'] += dS * omega_S[0]
                else:
                    self.sol['S'] += dS * omega_S[1]

            x = np.hstack((self.sol['psi'][1:-1],
                           self.sol['phi_n'][1:-1],
                           self.sol['phi_p'][1:-1],
                           np.array([self.sol['S']])))
            fluct += (l2_norm(dx) / l2_norm(x)) / nz

        self._fit_Sf_Sb()
        S = self._calculate_S()
        for i in range(self.nz):
            self.sol2d[i]['S'] = S[i]
        return fluct

    def _calculate_S(self, Sf=None, Sb=None):
        if Sf is None or Sb is None:
            assert self.ndim == 2
            Sf = self.Sf
            Sb = self.Sb
        S_calc = Sf + Sb
        f = interpolate.InterpolatedUnivariateSpline(self.zbn,
                                                    S_calc,
                                                    k=3)
        S = f(self.zin)
        # S = (S_calc[1:] + S_calc[:-1]) / 2
        return S
        # for i in range(self.nz):
        #     self.sol2d[i]['S'] = (S_calc[i] + S_calc[i+1]) / 2


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sample_design import sd

    # initialize problem
    ld = LaserDiode2D(design=sd, ar_inds=3,
                      L=3000e-4, w=100e-4,
                      R1=0.95, R2=0.05,
                      lam=0.87e-4, ng=3.9,
                      alpha_i=0.5, beta_sp=1e-4)
    ld.gen_nonuniform_mesh(param='Eg', y_ext=[0.3, 0.3])
    ld.make_dimensionless()
    ld.solve_waveguide()
    ld.solve_equilibrium()

    # solve 1D problem until there is significant photon density
    V = 0.0
    dV = 0.1
    while ld.sol['S'] < 1.0:
        print(V, end=', ')
        ld.lasing_init(V)
        fluct = 1.0
        while fluct > 1e-8:
            if fluct > 1e-3:
                omega = 0.05
            else:
                omega = 0.2
            fluct = ld.lasing_step(omega, (1., omega))
        print(ld.iterations)
        V += dV

    # convert to 2D and perform one iteration along longitudinal axis
    S_1D = ld.sol['S']
    ld.to_2D(10)

    S0 = np.array([d['S'] for d in ld.sol2d])
    flucts = list()
    for i in range(100):
        fluct = ld.lasing_step_2D(omega=0.25, omega_S=(0.25, 0.25), niter=10)
        flucts.append(fluct)
        print(i, fluct, ld.sol2d[0]['S'])
    S1 = np.array([d['S'] for d in ld.sol2d])

    plt.figure()
    plt.plot(S0)
    plt.plot(S1)

    plt.figure()
    plt.semilogy(flucts)
