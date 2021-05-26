# -*- coding: utf-8 -*-
"""
2-dimensional model of a diode laser.
"""

import numpy as np
from scipy import sparse
from ld_1d import LaserDiode1D
import units
from newton import l2_norm
import recombination as rec


class LaserDiode2D(LaserDiode1D):
    def __init__(self, design, ar_inds, L, w, R1, R2,
                 lam, ng, alpha_i, beta_sp):
        LaserDiode1D.__init__(self, design, ar_inds, L, w, R1, R2,
                              lam, ng, alpha_i, beta_sp)
        self.nz = 1          # number of z grid nodes
        self.dz = L          # longitudinal grid step
        self.sol2d = list()  # solution at every slice
        self.Sf = np.zeros(self.nz)
        self.Sb = np.zeros(self.nz)
        self.ndim = 1        # initialize as 1D

    def make_dimensionless(self):
        LaserDiode1D.make_dimensionless(self)
        self.dz /= units.x

    def original_units(self):
        LaserDiode1D.original_units()
        self.dz *= units.x

    def to_2D(self, n):
        self.nz = n
        self.sol2d = [dict() for _ in range(n)]
        S0 = self.sol['S'] / 2
        for i in range(n):
            for key in ('psi', 'phi_n', 'phi_p', 'n', 'p'):
                self.sol2d[i][key] = self.sol[key].copy()
        self.dz = self.L / n
        self.ndim = 2
        self._calculate_Sf_Sb(S0)
        for i in range(n):
            self.sol2d[i]['S'] = self.Sf[i] + self.Sb[i]

    def _calculate_Sf_Sb(self, S0):
        assert self.ndim == 2

        # calculate gain and radiative recombination rate
        ixa = self.ar_ix
        g = np.zeros(self.nz)
        R_rad = np.zeros(self.nz)
        for i in range(self.nz):
            self.sol = self.sol2d[i]

            # material gain
            n = self.sol['n'][ixa]
            p = self.sol['p'][ixa]
            N = np.zeros_like(n)
            ixn = n < p
            N[ixn] = n[ixn]
            N[~ixn] = p[~ixn]
            gain = self.yin['g0'][ixa] * np.log(N / self.yin['N_tr'][ixa])

            # modal gain
            self.sol = self.sol2d[i]
            alpha = self.alpha_i + self._calculate_fca()
            w = (self.xbn[1:] - self.xbn[:-1])[ixa[1:-1]]
            T = self.yin['wg_mode'][ixa]
            g[i] = np.sum(gain * T * w) - alpha

            # radiative recombination
            R = rec.rad_R(self.sol['n'][ixa], self.sol['p'][ixa],
                          self.yin['n0'][ixa], self.yin['p0'][ixa],
                          self.yin['B'][ixa])
            R_rad[i] = np.sum(R * T * w) / 2

        # calculate photon densities
        Sf = np.zeros(self.nz)
        Sb = np.zeros(self.nz)
        Sf[0] = S0
        k = self.beta_sp / self.vg
        for i in range(0, self.nz-1):
            Sf[i+1] = Sf[i] + ((Sf[i]*g[i] + k*R_rad[i]) * self.dz)
        Sb[-1] = (Sf[-1] + ((Sf[-1]*g[-1] + k*R_rad[-1]) * self.dz)) * self.R2
        for i in range(self.nz-1, 0, -1):
            Sb[i-1] = Sb[i] + ((Sb[i]*g[i] + k*R_rad[i]) * self.dz)

        self.Sf = Sf
        self.Sb = Sb

    def _transport_system_2D(self, Phi, discr):
        J, r = self._transport_system(discr, laser=True, save_J=False,
                                      save_Isp=False)
        r[-1] += self.vg*(Phi + self.sol['S'] * self.alpha_m)
        J[-1, -1] += self.vg * self.sol['S'] * self.alpha_m
        return J, r

    def lasing_step_2D(self, discr='mSG', niter=10, omega=0.1,
                       omega_S=(1.0, 0.1)):
        Sf = self.Sf
        Sb = self.Sb
        nz = len(self.sol2d)
        Phi = np.zeros(nz)
        Phi[1:-1] = (Sb[2:] - Sb[1:-1] - Sf[1:-1] + Sf[:-2]) / self.dz
        Phi[0] = Sb[1] - Sb[0] - Sf[0] + Sb[0] * self.R1
        Phi[-1] = Sf[-1] * self.R2 - Sb[-1] - Sf[-1] + Sf[-2]
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

        # self._calculate_Sf_Sb()
        return fluct


if __name__ == '__main__':
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
    S = ld.sol['S']
    ld.to_2D(10)
    fluct = ld.lasing_step_2D()
