# -*- coding: utf-8 -*-
"""
Tools for calculating photon density in a laser diode.
"""

import numpy as np
import carrier_concentrations as cc
import recombination as rec

def gain_rate(n, p, S, g0, N_tr, wg_mode, omega, vg):
    "Photon generation rate due to stimulated emission."
    stack = np.stack([n, p])
    nmin = np.min(stack, axis=0)
    gmat = g0*np.log(nmin/N_tr)
    R_st = vg*S*np.sum(gmat*wg_mode*omega)
    R_st = np.max([R_st, 0])  # ignoring absorption
    return R_st

def fca_loss(n, p, wg_mode, omega, fca_e, fca_h):
    "Calculate loss due to free-carrier absorption."
    alpha = n*fca_e+p*fca_h  # array
    alpha = np.sum(alpha*wg_mode*omega)  # weighted average
    return alpha

def loss_rate(n, p, S, vg, alpha_i, alpha_m, wg_mode, omega,
              fca_e, fca_h):
    "Photon density decrease rate due to internal absorption."
    alpha_fca = fca_loss(n, p, wg_mode, omega, fca_e, fca_h)
    R_loss = vg*(alpha_i+alpha_m+alpha_fca)*S
    return R_loss

def delta_S(psi, phi_n, phi_p, S, ld, delta_t=0.01):
    "Calculate change in photon density occuring over time interval `delta_t`."

    # carrier densities
    n = cc.n(psi, phi_n, ld.values['Nc'], ld.values['Ec'], ld.Vt)
    p = cc.p(psi, phi_p, ld.values['Nv'], ld.values['Ev'], ld.Vt)

    # volumes
    omega = np.zeros_like(ld.x)
    omega[1:-1] = ld.xm[1:]-ld.xm[:-1]
    omega[0] = ld.x[0]
    omega[-1] = ld.x[-1]-ld.x[-2]

    # photon rates
    R_st = gain_rate(n=n, p=p, S=S,
                     g0=ld.values['g0'],
                     N_tr=ld.values['N_tr'],
                     wg_mode=ld.values['wg_mode'],
                     omega=omega,
                     vg=ld.vg)
    R_sp_total = rec.rad_R(psi, phi_n, phi_p,
                           Nc=ld.values['Nc'],
                           Nv=ld.values['Nv'],
                           Ec=ld.values['Ec'],
                           Ev=ld.values['Ev'],
                           Vt=ld.Vt,
                           n0=ld.values['n0'],
                           p0=ld.values['p0'],
                           B=ld.values['B'])
    R_sp_modal = R_sp_total*ld.values['wg_mode']*omega
    R_sp = np.sum(R_sp_modal[ld.ar_ix])*ld.beta_sp*ld.x[-1]
    R_loss = loss_rate(n=n, p=p, S=S, vg=ld.vg,
                       alpha_i=ld.alpha_i, alpha_m=ld.alpha_m,
                       wg_mode=ld.values['wg_mode'],
                       omega=omega,
                       fca_e=ld.fca_e,
                       fca_h=ld.fca_h)
    dS = (R_st+R_sp-R_loss)*delta_t

    return dS
