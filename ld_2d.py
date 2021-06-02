# -*- coding: utf-8 -*-
"""
2-dimensional model of a diode laser.
"""

import numpy as np
from scipy import sparse
from ld_1d import LaserDiode1D
import units
import carrier_concentrations as cc
import flux
import recombination as rec
import vrs


class LaserDiode2D(LaserDiode1D):
    def __init__(self, design, ar_inds, L, w, R1, R2,
                 lam, ng, alpha_i, beta_sp):
        LaserDiode1D.__init__(self, design, ar_inds, L, w, R1, R2,
                              lam, ng, alpha_i, beta_sp)
        self.nz = 1                  # number of z grid nodes
        self.dz = L                  # longitudinal grid step
        self.zin = np.array(L/2)     # z grid nodes
        self.zbn = np.array([0, L])  # z grid volume boundaries
        self.sol2d = list()          # solution at every slice
        self.ndim = 1                # initialize as 1D

        # photon distribution along z
        self.Sf = np.zeros(self.nz)  # forward-propagating
        self.Sb = np.zeros(self.nz)  # backward-propagating

    def make_dimensionless(self):
        LaserDiode1D.make_dimensionless(self)
        self.dz /= units.x
        self.zin /= units.x
        self.zbn /= units.x
        self.Sf /= units.n * units.x
        self.Sb /= units.n * units.x

        # 2D solution
        for sol in self.sol2d:
            for key in sol:
                sol[key] /= units.dct[key]

    def original_units(self):
        LaserDiode1D.original_units(self)
        self.dz *= units.x
        self.zin *= units.x
        self.zbn *= units.x
        self.Sf *= units.n * units.x
        self.Sb *= units.n * units.x

        # 2D solution
        for sol in self.sol2d:
            for key in sol:
                sol[key] *= units.dct[key]

    def to_2D(self, n):
        self.nz = n
        self.ndim = 2
        self.dz = self.L / n
        self.zin = np.arange(self.dz / 2, self.L, self.dz)
        self.zbn = np.linspace(0, self.L, self.nz + 1)
        S = self.sol['S']
        self.Sf = np.zeros(n + 1)
        self.Sf[0] = S * self.R1 / (1 + self.R1)
        self.Sf[1:-1] = S / 2
        self.Sf[-1] = S / (1 + self.R2)
        self.Sb = np.zeros(n + 1)
        self.Sb[0] = S / (1 + self.R1)
        self.Sb[1:-1] = self.sol['S'] / 2
        self.Sb[-1] = S * self.R2 / (1 + self.R2)
        self.sol2d = [dict() for _ in range(n)]
        for i in range(n):
            for key in ('psi', 'phi_n', 'phi_p', 'n', 'p'):
                self.sol2d[i][key] = self.sol[key].copy()

    def _transport_system_2D(self, discr='mSG'):
        # mesh parameters
        m = self.npoints - 2
        h = self.xin[1:] - self.xin[:-1]
        w = self.xbn[1:] - self.xbn[:-1]
        nz = self.nz

        ixa = self.ar_ix  # mask for active region nodes
        inds_a = np.where(ixa)[0]
        nxa = np.sum(ixa)
        g0 = self.yin['g0'][ixa]
        N_tr = self.yin['N_tr'][ixa]
        T = self.yin['wg_mode'][ixa]
        w_ar = w[ixa[1:-1]]
        mSf = self.Sf
        Sf = (mSf[1:] + mSf[:-1]) / 2
        mSb = self.Sb
        Sb = (mSb[1:] + mSb[:-1]) / 2
        S = Sf + Sb

        # Jacobian
        data = np.zeros((11, 3*m*nz + nz*2))  # transport diagonals
        diags = [2*m, m, 1, 0, -1, -m+1, -m, -m-1, -2*m+1, -2*m, -2*m-1]
        J4_13 = np.zeros((2, nxa * 3 * nz))   # bottom rows
        J24 = np.zeros(nxa * nz)          # dF2 / dS
        J44 = np.zeros((nz * 2, nz * 2))      # dF4 / dS

        rvec = np.zeros(m * 3 * nz + nz * 2)  # vector of residuals

        for num in range(self.nz):
            self.sol = self.sol2d[num]

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

            # current densities and their derivatives
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
                                      p[1:-1], dp_dpsi[1:-1], B_rad)
            dRaug_dpsi = rec.auger_Rdot(n[1:-1], dn_dpsi[1:-1],
                                        p[1:-1], dp_dpsi[1:-1],
                                        n0, p0, Cn, Cp)
            dR_dpsi = dRsrh_dpsi + dRrad_dpsi + dRaug_dpsi
            dRsrh_dphin = rec.srh_Rdot(n[1:-1], dn_dphin[1:-1], p[1:-1], 0,
                                       n0, p0, tau_n, tau_p)
            dRrad_dphin = rec.rad_Rdot(n[1:-1], dn_dphin[1:-1], p[1:-1], 0,
                                       B_rad)
            dRaug_dphin = rec.auger_Rdot(n[1:-1], dn_dphin[1:-1],
                                         p[1:-1], 0,  n0, p0, Cn, Cp)
            dR_dphin = dRsrh_dphin + dRrad_dphin + dRaug_dphin
            dRsrh_dphip = rec.srh_Rdot(n[1:-1], 0, p[1:-1], dp_dphip[1:-1],
                                       n0, p0, tau_n, tau_p)
            dRrad_dphip = rec.rad_Rdot(n[1:-1], 0, p[1:-1], dp_dphip[1:-1],
                                       B_rad)
            dRaug_dphip = rec.auger_Rdot(n[1:-1], 0, p[1:-1],
                                         dp_dphip[1:-1], n0, p0, Cn, Cp)
            dR_dphip = dRsrh_dphip + dRrad_dphip + dRaug_dphip

            # gain
            ixn = (n[ixa] < p[ixa])
            N = n[ixa].copy()
            N[~ixn] = p[ixa][~ixn]
            gain = g0 * np.log(N / N_tr)
            ix_abs = np.where(gain < 0)
            gain[ix_abs] = 0.0  # ignore absorption

            # gain derivatives
            gain_dpsi = np.zeros_like(gain)
            gain_dpsi[ixn] = g0[ixn] * dn_dpsi[ixa][ixn] / n[ixa][ixn]
            gain_dpsi[~ixn] = g0[~ixn] * dp_dpsi[ixa][~ixn] / p[ixa][~ixn]
            gain_dphin = np.zeros_like(gain)
            gain_dphin[ixn] = g0[ixn] * dn_dphin[ixa][ixn] / n[ixa][ixn]
            gain_dphip = np.zeros_like(gain)
            gain_dphip[~ixn] = g0[~ixn] * dp_dphip[ixa][~ixn] / p[ixa][~ixn]
            for gdot in [gain_dpsi, gain_dphin, gain_dphip]:
                gdot[ix_abs] = 0  # ignore absoption

            # net gain
            alpha_fca = self._calculate_fca()
            alpha = self.alpha_i + alpha_fca
            net_gain = np.sum(gain * w_ar * T) - alpha

            # stimulated recombination rate
            R_st = self.vg * gain * w_ar * T * S[num]
            dRst_dS = self.vg * gain * w_ar * T / 2
            dRst_dpsi = self.vg * gain_dpsi * w_ar * T * S[num]
            dRst_dphin = self.vg * gain_dphin * w_ar * T * S[num]
            dRst_dphip = self.vg * gain_dphip * w_ar * T * S[num]

            # residual
            rj = rvec[(3*m*num):(3*m*(num+1))]
            rj[0:m] = vrs.poisson_res(psi, n, p, h, w, self.yin['eps'],
                                      self.eps_0, self.q, self.yin['C_dop'])
            rj[m:2*m] = self.q*R*w - (jn[1:]-jn[:-1])
            rj[m:2*m][ixa[1:-1]] += self.q * R_st
            rj[2*m:3*m] = -self.q*R*w - (jp[1:]-jp[:-1])
            rj[2*m:3*m][ixa[1:-1]] -= self.q * R_st
            R_rad_ar = np.sum(R_rad[ixa[1:-1]]*w_ar*T) / 2
            rvec[3*m*nz + num] = \
                (self.vg*(mSb[num+1] - mSb[num]) / self.dz
                 + self.vg * net_gain * Sb[num]
                 + self.beta_sp * R_rad_ar)
            rvec[3*m*nz + nz + num] = \
                (-self.vg*(mSf[num+1] - mSf[num]) / self.dz
                 + self.vg * net_gain * Sf[num]
                 + self.beta_sp * R_rad_ar)

            # Jacobian
            # 1. Poisson's equation
            j11 = vrs.poisson_dF_dpsi(dn_dpsi, dp_dpsi, h, w,
                                      self.yin['eps'], self.eps_0, self.q)
            j12 = vrs.poisson_dF_dphin(dn_dphin, w, self.eps_0, self.q)
            j13 = vrs.poisson_dF_dphip(dp_dphip, w, self.eps_0, self.q)

            # 2. Electron current continuity equation
            j21 = vrs.jn_dF_dpsi(djn_dpsi1, djn_dpsi2, dR_dpsi, w,
                                 self.q, m)
            j21[1, ixa[1:-1]] += self.q * dRst_dpsi
            j22 = vrs.jn_dF_dphin(djn_dphin1, djn_dphin2, dR_dphin,
                                  w, self.q, m)
            j22[1, ixa[1:-1]] += self.q * dRst_dphin
            j23 = vrs.jn_dF_dphip(dR_dphip, w, self.q, m)
            j23[ixa[1:-1]] += self.q * dRst_dphip
            J24[num*nxa:(num+1)*nxa] = self.q * dRst_dS

            # 3. Hole current continuity equation
            j31 = vrs.jp_dF_dpsi(djp_dpsi1, djp_dpsi2, dR_dpsi,
                                 w, self.q, m)
            j31[1, ixa[1:-1]] -= self.q * dRst_dpsi
            j32 = vrs.jp_dF_dphin(dR_dphin, w, self.q, m)
            j32[ixa[1:-1]] -= self.q * dRst_dphin
            j33 = vrs.jp_dF_dphip(djp_dphip1, djp_dphip2, dR_dphip,
                                  w, self.q, m)
            j33[1, ixa[1:-1]] -= self.q * dRst_dphip
            # dF3/dS = -dF2/dS

            # 4. Rate equations for densities of backward-
            # and forward-propagating photons
            J413_j = J4_13[:, num*nxa*3:(num+1)*nxa*3]
            J413_j[0, :nxa] = \
                (self.beta_sp * dRrad_dpsi[ixa[1:-1]] * w_ar * T
                 + self.vg * gain_dpsi * w_ar * T * Sb[num])
            J413_j[1, :nxa] = \
                (self.beta_sp * dRrad_dpsi[ixa[1:-1]] * w_ar * T
                 + self.vg * gain_dpsi * w_ar * T * Sf[num])
            J413_j[0, nxa:2*nxa] = \
                (self.beta_sp * dRrad_dphin[ixa[1:-1]] * w_ar * T
                 + self.vg * gain_dphin * w_ar * T * Sb[num])
            J413_j[1, nxa:2*nxa] = \
                (self.beta_sp * dRrad_dphin[ixa[1:-1]] * w_ar * T
                 + self.vg * gain_dphin * w_ar * T * Sf[num])
            J413_j[0, 2*nxa:3*nxa] = \
                (self.beta_sp * dRrad_dphin[ixa[1:-1]] * w_ar * T
                 + self.vg * gain_dphip * w_ar * T * Sb[num])
            J413_j[1, 2*nxa:3*nxa] = \
                (self.beta_sp * dRrad_dphip[ixa[1:-1]] * w_ar * T
                 + self.vg * gain_dphip * w_ar * T * Sf[num])

            J44[num, num] = self.vg * (-1 / self.dz + net_gain / 2)
            if num < nz - 1:
                J44[num, num+1] = self.vg * (1 / self.dz + net_gain / 2)
            else:
                J44[nz-1, -1] = self.vg * self.R2 * (1 / self.dz
                                                   + net_gain / 2)

            J44[nz + num, nz + num] = self.vg * (-1 / self.dz
                                                 + net_gain / 2)
            if num > 0:
                J44[nz + num, nz + num-1] = self.vg * (1 / self.dz
                                                       + net_gain / 2)
            else:
                J44[nz, 0] = self.vg * self.R1 * (1 / self.dz
                                                  + net_gain / 2)

            # collect Jacobian diagonals
            data_j = data[:, (3*m*num):(3*m*(num+1))]
            data_j[0, 2*m:   ] = j13
            data_j[1,   m:2*m] = j12
            data_j[1, 2*m:   ] = j23
            data_j[2,    :m  ] = j11[0]
            data_j[2,   m:2*m] = j22[0]
            data_j[2, 2*m:   ] = j33[0]
            data_j[3,    :m  ] = j11[1]
            data_j[3,   m:2*m] = j22[1]
            data_j[3, 2*m:   ] = j33[1]
            data_j[4,    :m  ] = j11[2]
            data_j[4,   m:2*m] = j22[2]
            data_j[4, 2*m:   ] = j33[2]
            data_j[5,    :m  ] = j21[0]
            data_j[6,    :m  ] = j21[1]
            data_j[6,   m:2*m] = j32
            data_j[7,    :m  ] = j21[2]
            data_j[8,    :m  ] = j31[0]
            data_j[9,    :m  ] = j31[1]
            data_j[10,   :m  ] = j31[2]

        # assemble Jacobian
        J = sparse.spdiags(data, diags, format='lil',
                           m=(3*m)*nz + nz*2, n=(3*m)*nz + nz*2)

        # rightmost columns, first slice
        J[m + inds_a, 3*m*nz] = J24[:nxa] * (1 + self.R1)
        J[2*m + inds_a, 3*m*nz] = -J24[:nxa] * (1 + self.R1)
        J[m + inds_a, 3*m*nz + 1] = J24[:nxa]
        J[2*m + inds_a, 3*m*nz + 1] = -J24[:nxa]
        J[m + inds_a, 3*m*nz + nz] = J24[:nxa]
        J[2*m + inds_a, 3*m*nz + nz] = -J24[:nxa]

        # rightmost columns, inner slices
        for num in range(1, nz-1):
            q_Rstdot = J24[nxa*num:nxa*(num+1)]  # q*dRst/dSb
            for k in [0, 1, nz-1, nz]:
                J[num*m*3 + m + inds_a, 3*m*nz + num+k] = q_Rstdot
                J[num*m*3 + 2*m + inds_a, 3*m*nz + num+k] = -q_Rstdot

        # rightmost columns, last slice
        J[(nz-1)*3*m + m + inds_a, 3*m*nz + nz-1] = J24[-nxa:]
        J[(nz-1)*3*m + 2*m + inds_a, 3*m*nz + nz-1] = -J24[-nxa:]
        J[(nz-1)*3*m + m + inds_a, -2] = J24[-nxa:]
        J[(nz-1)*3*m + 2*m + inds_a, -2] = -J24[-nxa:]
        J[(nz-1)*3*m + m + inds_a, -1] = J24[-nxa:] * (1 + self.R2)
        J[(nz-1)*3*m + 2*m + inds_a, -1] = -J24[-nxa:] * (1 + self.R2)

        # bottom rows
        for num in range(nz):
            for k in range(3):
                J[3*m*nz + num, num*(3*m) + k*m + inds_a] = \
                    J4_13[0, num*(3*nxa) + k*nxa:num*(3*nxa) + (k+1)*nxa]
                J[3*m*nz + nz + num, num*(3*m) + k*m + inds_a] = \
                    J4_13[1, num*(3*nxa) + k*nxa:num*(3*nxa) + (k+1)*nxa]

        J[nz*(3*m):, nz*(3*m):] = J44
        J = J.tocsc()

        self.sol = dict()

        return J, rvec

    def lasing_step_2D(self, omega=0.1, omega_S=(1.0, 0.1), discr='mSG'):
        J, r = self._transport_system_2D(discr)
        dx = sparse.linalg.spsolve(J, -r)
        m = self.npoints - 2
        for j, sol in enumerate(self.sol2d):
            dx_j = dx[3*m*j:3*m*(j+1)]
            sol['psi'][1:-1] += omega * dx_j[:m]
            sol['phi_n'][1:-1] += omega * dx_j[m:2*m]
            sol['phi_p'][1:-1] += omega * dx_j[2*m:3*m]
            sol['n'] = cc.n(sol['psi'], sol['phi_n'],
                            self.yin['Nc'], self.yin['Ec'], self.Vt)
            sol['p'] = cc.p(sol['psi'], sol['phi_p'],
                            self.yin['Nv'], self.yin['Ev'], self.Vt)
        delta_Sb = dx[-2*self.nz:-self.nz]
        ixb = delta_Sb > 0
        delta_Sf = dx[-self.nz:]
        ixf = delta_Sf > 0

        self.Sb[:-1][ixb] += omega_S[0] * delta_Sb[ixb]
        self.Sb[:-1][~ixb] += omega_S[1] * delta_Sb[~ixb]
        self.Sf[0] = self.Sb[0] * self.R1
        self.Sf[1:][ixf] += omega_S[0] * delta_Sf[ixf]
        self.Sf[1:][~ixf] += omega_S[1] * delta_Sf[~ixf]
        self.Sb[-1] = self.Sf[-1] * self.R2

        return dx


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sample_design import sd

    plt.rcdefaults()
    plt.rc('lines', linewidth=0.8)

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
    while ld.sol['S'] < 1:
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

    # convert to 2D
    S_1D = ld.sol['S']
    ld.to_2D(10)
    ld.original_units()

    # plot initial (constant) S(z) curve
    plt.figure(1)
    plt.plot(ld.zbn*1e4, ld.Sf+ld.Sb, 'k:.', label='$S_{1D}$')
    plt.xlabel(r'$z$ ($\mu$m)')
    plt.ylabel('$S$ (cm$^{-2}$)')

    # plot band diagram
    plt.figure(2)
    x = ld.xin * 1e4
    c = 'k'
    plt.plot(x, ld.yin['Ec']-ld.sol['psi'], color=c, ls='-', label='1D')
    plt.plot(x, ld.yin['Ev']-ld.sol['psi'], color=c, ls='-')
    plt.plot(x, -ld.sol['phi_n'], color=c, ls=':')
    plt.plot(x, -ld.sol['phi_p'], color=c, ls=':')
    plt.xlabel(r'$x$ ($\mu$m)')
    plt.ylabel('$E$ (eV)')

    # solve 2D problem
    ld.make_dimensionless()
    print('Solving 2D problem...')
    n_iter = 20
    for i in range(n_iter):
        dx = ld.lasing_step_2D(1.0, (1.0, 1.0))
        print(f'{i+1} / {n_iter}')
    ld.original_units()

    # plot calculated photon density distributions
    plt.figure(1)
    plt.plot(ld.zbn*1e4, ld.Sf, 'b-x', label='$S_f$')
    plt.plot(ld.zbn*1e4, ld.Sb, 'r-x', label='$S_b$')
    plt.plot(ld.zbn*1e4, ld.Sf+ld.Sb, 'k-x', label='$S_{2D}$')
    plt.legend()

    # plot band diagrams at the opposite laser edges
    plt.figure(2)
    labels = ['2D, $z = 0$', '2D, $z = L$']
    for j, c, label in zip([0, 9], ['b', 'r'], labels):
        psi = ld.sol2d[j]['psi']
        plt.plot(x, ld.yin['Ec']-psi, color=c, ls='-', label=label)
        plt.plot(x, ld.yin['Ev']-psi, color=c, ls='-')
        plt.plot(x, -ld.sol2d[j]['phi_n'], color=c, ls=':')
        plt.plot(x, -ld.sol2d[j]['phi_p'], color=c, ls=':')
        plt.legend()
