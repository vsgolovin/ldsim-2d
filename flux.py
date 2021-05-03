# -*- coding: utf-8 -*-
"""
Current density calculation.
"""

import numpy as np
import sdf

def bernoulli(x):
    if isinstance(x, np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.true_divide(x, (np.exp(x)-1))
            y[np.where(np.abs(x)<1e-12)] = 1
    elif np.abs(x)<1e-12:
        y = 1
    else:
        y = x/(np.exp(x)-1)
    return y

def bernoulli_dot(x):
    if isinstance(x, np.ndarray):
        enum = np.exp(x)-x*np.exp(x)-1
        denom = (np.exp(x)-1)**2
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.true_divide(enum, denom)
            y[np.where(np.abs(x)<1e-12)] = -0.5
    elif np.abs(x)<1e-12:
        y = -0.5
    else:
        y = (np.exp(x)-x*np.exp(x)-1) / (np.exp(x)-1)**2
    return y

#%% Scharfetter-Gummel expressions for current density and its derivatives
def SG_jn(n1, n2, B_plus, B_minus, h, Vt, q, mu_n):
    "Scharfetter-Gummel formula for electron current density."
    j = -q*mu_n*Vt/h * (n1*B_minus-n2*B_plus)
    return j

def SG_djn_dpsi1(n1, n2, ndot_1, B_minus, Bdot_plus, Bdot_minus, h,
                 Vt, q, mu_n):
    jdot = -q*mu_n/h * (Bdot_minus*n1 + B_minus*ndot_1*Vt + Bdot_plus*n2)
    return jdot

def SG_djn_dpsi2(n1, n2, ndot_2, B_plus, Bdot_plus, Bdot_minus, h,
                 Vt, q, mu_n):
    jdot = q*mu_n/h * (Bdot_minus*n1 + Bdot_plus*n2 + B_plus*ndot_2*Vt)
    return jdot

def SG_djn_dphin1(ndot, B_minus, h, Vt, q, mu_n):
    jdot = -q*mu_n*Vt/h * B_minus * ndot
    return jdot

def SG_djn_dphin2(ndot, B_plus, h, Vt, q, mu_n):
    jdot = q*mu_n*Vt/h * B_plus * ndot
    return jdot

def SG_jp(p1, p2, B_plus, B_minus, h, Vt, q, mu_p):
    "Scharfetter-Gummel formula for hole current density."
    j =  q*mu_p*Vt/h * (p1*B_plus - p2*B_minus)
    return j

def SG_djp_dpsi1(p1, p2, pdot_1, B_plus, Bdot_plus, Bdot_minus, h,
                 Vt, q, mu_p):
    jdot = q*mu_p/h * (-Bdot_plus*p1+B_plus*pdot_1*Vt-Bdot_minus*p2)
    return jdot

def SG_djp_dpsi2(p1, p2, pdot_2, B_minus, Bdot_plus, Bdot_minus, h,
                 Vt, q, mu_p):
    jdot = q*mu_p/h * (Bdot_plus*p1+Bdot_minus*p2-B_minus*pdot_2*Vt)
    return jdot

def SG_djp_dphip1(pdot, B_plus, h, Vt, q, mu_p):
    jdot = q*mu_p*Vt/h * B_plus * pdot
    return jdot

def SG_djp_dphip2(pdot, B_minus, h, Vt, q, mu_p):
    jdot = -q*mu_p*Vt/h * B_minus * pdot
    return jdot


#%% Modified Scharfetter-Gummel scheme
def mSG_g(nu_1, nu_2):
    "Diffusion enhancement factor."
    F = sdf.fermi_fdint
    g = np.sqrt( (F(nu_1)*F(nu_2)) / (np.exp(nu_1)*np.exp(nu_2)) )
    return g

def mSG_gdot(nu_1, nu_2):
    "Diffusion enhancement factor derivative with respect to nu_1."
    F = sdf.fermi_fdint
    Fdot = sdf.fermi_dot_fdint
    gdot = mSG_g(nu_1, nu_2)/2 * (Fdot(nu_1)/F(nu_1) - 1)
    return gdot

def mSG_jn(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n):
    "Electron current density."
    nu_1 = (psi_1-phi_n1-Ec) / Vt
    nu_2 = (psi_2-phi_n2-Ec) / Vt
    g = mSG_g(nu_1, nu_2)
    j = g*SG_jn(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    return j

def mSG_djn_dpsi1(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n):
    j_SG = SG_jn(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    jdot_SG = SG_djn_dpsi1(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q,
                           mu_n)
    nu_1 = (psi_1-phi_n1-Ec) / Vt
    nu_2 = (psi_2-phi_n2-Ec) / Vt
    g = mSG_g(nu_1, nu_2)
    gdot = mSG_gdot(nu_1, nu_2) * 1/Vt
    jdot = gdot*j_SG + g*jdot_SG
    return jdot

def mSG_djn_dpsi2(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n):
    j_SG = SG_jn(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    jdot_SG = SG_djn_dpsi2(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q,
                           mu_n)
    nu_1 = (psi_1-phi_n1-Ec) / Vt
    nu_2 = (psi_2-phi_n2-Ec) / Vt
    g = mSG_g(nu_1, nu_2)
    gdot = mSG_gdot(nu_2, nu_1) * 1/Vt
    jdot = gdot*j_SG + g*jdot_SG
    return jdot

def mSG_djn_dphin1(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n):
    j_SG = SG_jn(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    jdot_SG = SG_djn_dphin1(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt,
                            q, mu_n)
    nu_1 = (psi_1-phi_n1-Ec) / Vt
    nu_2 = (psi_2-phi_n2-Ec) / Vt
    g = mSG_g(nu_1, nu_2)
    gdot = mSG_gdot(nu_1, nu_2) * (-1)/Vt
    jdot = gdot*j_SG + g*jdot_SG
    return jdot

def mSG_djn_dphin2(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n):
    j_SG = SG_jn(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    jdot_SG = SG_djn_dphin2(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt,
                            q, mu_n)
    nu_1 = (psi_1-phi_n1-Ec) / Vt
    nu_2 = (psi_2-phi_n2-Ec) / Vt
    g = mSG_g(nu_1, nu_2)
    gdot = mSG_gdot(nu_2, nu_1) * (-1)/Vt
    jdot = gdot*j_SG + g*jdot_SG
    return jdot

def mSG_jp(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p):
    "Hole current density."
    nu_1 = (-psi_1+phi_p1+Ev) / Vt
    nu_2 = (-psi_2+phi_p2+Ev) / Vt
    g = mSG_g(nu_1, nu_2)
    j = g*SG_jp(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    return j

def mSG_djp_dpsi1(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p):
    nu_1 = (-psi_1+phi_p1+Ev) / Vt
    nu_2 = (-psi_2+phi_p2+Ev) / Vt
    j_SG = SG_jp(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    jdot_SG = SG_djp_dpsi1(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt,
                        q, mu_p)
    g = mSG_g(nu_1, nu_2)
    gdot = mSG_gdot(nu_1, nu_2) * (-1)/Vt
    jdot = gdot*j_SG + g*jdot_SG
    return jdot

def mSG_djp_dpsi2(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p):
    nu_1 = (-psi_1+phi_p1+Ev) / Vt
    nu_2 = (-psi_2+phi_p2+Ev) / Vt
    j_SG = SG_jp(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    jdot_SG = SG_djp_dpsi2(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt,
                           q, mu_p)
    g = mSG_g(nu_1, nu_2)
    gdot = mSG_gdot(nu_2, nu_1) * (-1)/Vt
    jdot = gdot*j_SG + g*jdot_SG
    return jdot

def mSG_djp_dphip1(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p):
    nu_1 = (-psi_1+phi_p1+Ev) / Vt
    nu_2 = (-psi_2+phi_p2+Ev) / Vt
    j_SG = SG_jp(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    jdot_SG = SG_djp_dphip1(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt,
                            q, mu_p)
    g = mSG_g(nu_1, nu_2)
    gdot = mSG_gdot(nu_1, nu_2) * 1/Vt
    jdot = gdot*j_SG + g*jdot_SG
    return jdot

def mSG_djp_dphip2(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p):
    nu_1 = (-psi_1+phi_p1+Ev) / Vt
    nu_2 = (-psi_2+phi_p2+Ev) / Vt
    j_SG = SG_jp(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    jdot_SG = SG_djp_dphip2(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt,
                            q, mu_p)
    g = mSG_g(nu_1, nu_2)
    gdot = mSG_gdot(nu_2, nu_1) * 1/Vt
    jdot = gdot*j_SG + g*jdot_SG
    return jdot

#%% testing
test_err = 1e-3

def test_SG_jn_derivatives():
    psi_1 = -0.1
    psi_2 =  0.1
    phi_n1 = 0.2
    phi_n2 = 0.32
    Ec = 4
    x1 = 1
    x2 = 2
    Nc = 140
    Vt = 2
    q = 1.2
    mu_n = 14.2

    # psi_1
    psi_1_1 = psi_1*0.999
    psi_1_2 = psi_1*1.001
    jn_1 = SG_jn(psi_1_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    jn_2 = SG_jn(psi_1_2, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    jn_dot_fd = (jn_2-jn_1) / (psi_1_2-psi_1_1)
    jn_dot = SG_djn_dpsi1(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    err_psi1 = abs(1-jn_dot/jn_dot_fd)

    # psi_2
    psi_2_1 = psi_2*0.999
    psi_2_2 = psi_2*1.001
    jn_1 = SG_jn(psi_1, psi_2_1, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    jn_2 = SG_jn(psi_1, psi_2_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    jn_dot_fd = (jn_2-jn_1) / (psi_2_2-psi_2_1)
    jn_dot = SG_djn_dpsi2(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    err_psi2 = abs(1-jn_dot/jn_dot_fd)

    # phi_n1
    phi_n1_1 = phi_n1*0.999
    phi_n1_2 = phi_n1*1.001
    jn_1 = SG_jn(psi_1, psi_2, phi_n1_1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    jn_2 = SG_jn(psi_1, psi_2, phi_n1_2, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    jn_dot_fd = (jn_2-jn_1) / (phi_n1_2-phi_n1_1)
    jn_dot = SG_djn_dphin1(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    err_phin1 = abs(1-jn_dot/jn_dot_fd)

    # phi_n2
    phi_n2_1 = phi_n2*0.999
    phi_n2_2 = phi_n2*1.001
    jn_1 = SG_jn(psi_1, psi_2, phi_n1, phi_n2_1, x1, x2, Nc, Ec, Vt, q, mu_n)
    jn_2 = SG_jn(psi_1, psi_2, phi_n1, phi_n2_2, x1, x2, Nc, Ec, Vt, q, mu_n)
    jn_dot_fd = (jn_2-jn_1) / (phi_n2_2-phi_n2_1)
    jn_dot = SG_djn_dphin2(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    err_phin2 = abs(1-jn_dot/jn_dot_fd)

    err = np.array([err_psi1, err_psi2, err_phin1, err_phin2])
    assert (err<test_err).all()

def test_SG_jp_derivatives():
    psi_1 = 0.3
    psi_2 =  0.21
    phi_p1 = 0.17
    phi_p2 = 0.24
    Ev = -3
    x1 = 1
    x2 = 2
    Nv = 200
    Vt = 1.1
    q = 0.9
    mu_p = 6.7

    # psi_1
    psi_1_1 = psi_1*0.999
    psi_1_2 = psi_1*1.001
    jp_1 = SG_jp(psi_1_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    jp_2 = SG_jp(psi_1_2, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    jp_dot_fd = (jp_2-jp_1) / (psi_1_2-psi_1_1)
    jp_dot = SG_djp_dpsi1(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    err_psi1 = abs(1-jp_dot/jp_dot_fd)

    # psi_2
    psi_2_1 = psi_2*0.999
    psi_2_2 = psi_2*1.001
    jp_1 = SG_jp(psi_1, psi_2_1, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    jp_2 = SG_jp(psi_1, psi_2_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    jp_dot_fd = (jp_2-jp_1) / (psi_2_2-psi_2_1)
    jp_dot = SG_djp_dpsi2(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    err_psi2 = abs(1-jp_dot/jp_dot_fd)

    # phi_p1
    phi_p1_1 = phi_p1*0.999
    phi_p1_2 = phi_p1*1.001
    jp_1 = SG_jp(psi_1, psi_2, phi_p1_1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    jp_2 = SG_jp(psi_1, psi_2, phi_p1_2, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    jp_dot_fd = (jp_2-jp_1) / (phi_p1_2-phi_p1_1)
    jp_dot = SG_djp_dphip1(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    err_phip1 = abs(1-jp_dot/jp_dot_fd)

    # phi_p2
    phi_p2_1 = phi_p2*0.999
    phi_p2_2 = phi_p2*1.001
    jp_1 = SG_jp(psi_1, psi_2, phi_p1, phi_p2_1, x1, x2, Nv, Ev, Vt, q, mu_p)
    jp_2 = SG_jp(psi_1, psi_2, phi_p1, phi_p2_2, x1, x2, Nv, Ev, Vt, q, mu_p)
    jp_dot_fd = (jp_2-jp_1) / (phi_p2_2-phi_p2_1)
    jp_dot = SG_djp_dphip2(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    err_phip2 = abs(1-jp_dot/jp_dot_fd)

    err = np.array([err_psi1, err_psi2, err_phip1, err_phip2])
    assert (err<test_err).all()

def test_mSG_jn_derivatives():
    psi_1 = -0.1
    psi_2 =  0.1
    phi_n1 = -1.2
    phi_n2 = -0.65
    Ec = 2
    x1 = 4
    x2 = 5
    Nc = 8.3
    Vt = 1.3
    q = 0.1
    mu_n = 9.72

    # psi_1
    psi_1_1 = psi_1*0.999
    psi_1_2 = psi_1*1.001
    jn_1 = mSG_jn(psi_1_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    jn_2 = mSG_jn(psi_1_2, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    jn_dot_fd = (jn_2-jn_1) / (psi_1_2-psi_1_1)
    jn_dot = mSG_djn_dpsi1(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    err_psi1 = abs(1-jn_dot/jn_dot_fd)

    # psi_2
    psi_2_1 = psi_2*0.999
    psi_2_2 = psi_2*1.001
    jn_1 = mSG_jn(psi_1, psi_2_1, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    jn_2 = mSG_jn(psi_1, psi_2_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    jn_dot_fd = (jn_2-jn_1) / (psi_2_2-psi_2_1)
    jn_dot = mSG_djn_dpsi2(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    err_psi2 = abs(1-jn_dot/jn_dot_fd)

    # phi_n1
    phi_n1_1 = phi_n1*0.999
    phi_n1_2 = phi_n1*1.001
    jn_1 = mSG_jn(psi_1, psi_2, phi_n1_1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    jn_2 = mSG_jn(psi_1, psi_2, phi_n1_2, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    jn_dot_fd = (jn_2-jn_1) / (phi_n1_2-phi_n1_1)
    jn_dot = mSG_djn_dphin1(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    err_phin1 = abs(1-jn_dot/jn_dot_fd)

    # phi_n2
    phi_n2_1 = phi_n2*0.999
    phi_n2_2 = phi_n2*1.001
    jn_1 = mSG_jn(psi_1, psi_2, phi_n1, phi_n2_1, x1, x2, Nc, Ec, Vt, q, mu_n)
    jn_2 = mSG_jn(psi_1, psi_2, phi_n1, phi_n2_2, x1, x2, Nc, Ec, Vt, q, mu_n)
    jn_dot_fd = (jn_2-jn_1) / (phi_n2_2-phi_n2_1)
    jn_dot = mSG_djn_dphin2(psi_1, psi_2, phi_n1, phi_n2, x1, x2, Nc, Ec, Vt, q, mu_n)
    err_phin2 = abs(1-jn_dot/jn_dot_fd)

    assert (np.array([err_psi1, err_psi2, err_phin1, err_phin2]) < test_err).all()

def test_mSG_jp_derivatives():
    psi_1 = -0.4
    psi_2 = -0.381
    phi_p1 = 1.17
    phi_p2 = 1.24
    Ev = -4.2
    x1 = 3
    x2 = 3.3
    Nv = 170
    Vt = 3 
    q = 2.1
    mu_p = 6.9

    # psi_1
    psi_1_1 = psi_1*0.999
    psi_1_2 = psi_1*1.001
    jp_1 = mSG_jp(psi_1_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    jp_2 = mSG_jp(psi_1_2, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    jp_dot_fd = (jp_2-jp_1) / (psi_1_2-psi_1_1)
    jp_dot = mSG_djp_dpsi1(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    err_psi1 = abs(1-jp_dot/jp_dot_fd)

    # psi_2
    psi_2_1 = psi_2*0.999
    psi_2_2 = psi_2*1.001
    jp_1 = mSG_jp(psi_1, psi_2_1, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    jp_2 = mSG_jp(psi_1, psi_2_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    jp_dot_fd = (jp_2-jp_1) / (psi_2_2-psi_2_1)
    jp_dot = mSG_djp_dpsi2(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    err_psi2 = abs(1-jp_dot/jp_dot_fd)

    # phi_p1
    phi_p1_1 = phi_p1*0.999
    phi_p1_2 = phi_p1*1.001
    jp_1 = mSG_jp(psi_1, psi_2, phi_p1_1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    jp_2 = mSG_jp(psi_1, psi_2, phi_p1_2, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    jp_dot_fd = (jp_2-jp_1) / (phi_p1_2-phi_p1_1)
    jp_dot = mSG_djp_dphip1(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    err_phip1 = abs(1-jp_dot/jp_dot_fd)

    # phi_p2
    phi_p2_1 = phi_p2*0.999
    phi_p2_2 = phi_p2*1.001
    jp_1 = mSG_jp(psi_1, psi_2, phi_p1, phi_p2_1, x1, x2, Nv, Ev, Vt, q, mu_p)
    jp_2 = mSG_jp(psi_1, psi_2, phi_p1, phi_p2_2, x1, x2, Nv, Ev, Vt, q, mu_p)
    jp_dot_fd = (jp_2-jp_1) / (phi_p2_2-phi_p2_1)
    jp_dot = mSG_djp_dphip2(psi_1, psi_2, phi_p1, phi_p2, x1, x2, Nv, Ev, Vt, q, mu_p)
    err_phip2 = abs(1-jp_dot/jp_dot_fd)

    assert (np.array([err_psi1, err_psi2, err_phip1, err_phip2]) < test_err).all()

