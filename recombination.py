# -*- coding: utf-8 -*-

"""
A collection of functions for calculating recombination rates and corresponding
derivatives.
"""
import carrier_concentrations as cc

def srh_R(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, ni, n0, p0, tau_n, tau_p):
    """
    Shockley-Read-Hall recombination rate.
    """
    n = cc.n(psi, phi_n, Nc, Ec, Vt)
    p = cc.p(psi, phi_p, Nv, Ev, Vt)
    R = (n*p - n0*p0) / ((n+ni)*tau_p + (p+ni)*tau_n)
    return R

def srh_derivative(n, ndot, p, pdot, ni, n0, p0, tau_n, tau_p):
    """
    Auxillary function for repetative calculations.
    """
    u = n*p - n0*p0
    v = (n+ni)*tau_p + (p+ni)*tau_n
    udot = ndot*p + n*pdot
    vdot = tau_p*ndot + tau_n*pdot
    Rdot = (udot*v - u*vdot) / v**2
    return Rdot

def srh_dR_dpsi(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, ni, n0, p0,
                tau_n, tau_p):
    """
    Shockley-Read-Hall recombination rate derivative with respect to potential.
    """
    n = cc.n(psi, phi_n, Nc, Ec, Vt)
    ndot = cc.dn_dpsi(psi, phi_n, Nc, Ec, Vt)
    p = cc.p(psi, phi_p, Nv, Ev, Vt)
    pdot = cc.dp_dpsi(psi, phi_p, Nv, Ev, Vt)
    Rdot = srh_derivative(n, ndot, p, pdot, ni, n0, p0, tau_n, tau_p)
    return Rdot

def srh_dR_dphin(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, ni, n0, p0,
                 tau_n, tau_p):
    """
    Shockley-Read-Hall recombination rate derivative with respect to electron
    quasi Fermi potential.
    """
    n = cc.n(psi, phi_n, Nc, Ec, Vt)
    ndot = cc.dn_dphin(psi, phi_n, Nc, Ec, Vt)
    p = cc.p(psi, phi_p, Nv, Ev, Vt)
    pdot = 0
    Rdot = srh_derivative(n, ndot, p, pdot, ni, n0, p0, tau_n, tau_p)
    return Rdot

def srh_dR_dphip(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, ni, n0, p0,
                 tau_n, tau_p):
    """
    Shockley-Read-Hall recombination rate with respect to hole quasi Fermi
    potential.
    """
    n = cc.n(psi, phi_n, Nc, Ec, Vt)
    ndot = 0
    p = cc.p(psi, phi_p, Nv, Ev, Vt)
    pdot = cc.dp_dphip(psi, phi_p, Nv, Ev, Vt)
    Rdot = srh_derivative(n, ndot, p, pdot, ni, n0, p0, tau_n, tau_p)
    return Rdot

def rad_R(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, B):
    """
    Radiative recombination rate.
    """
    n = cc.n(psi, phi_n, Nc, Ec, Vt)
    p = cc.p(psi, phi_p, Nv, Ev, Vt)
    R = B*(n*p-n0*p0)
    return R

def rad_dR_dpsi(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, B):
    """
    Radiative recombination rate derivative with respect to potential.
    """
    n = cc.n(psi, phi_n, Nc, Ec, Vt)
    ndot = cc.dn_dpsi(psi, phi_n, Nc, Ec, Vt)
    p = cc.p(psi, phi_p, Nv, Ev, Vt)
    pdot = cc.dp_dpsi(psi, phi_p, Nv, Ev, Vt)
    Rdot = B*(n*pdot+p*ndot)
    return Rdot

def rad_dR_dphin(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, B):
    """
    Radiative recombination rate derivative with respect to electron quasi
    Fermi potential.
    """
    ndot = cc.dn_dphin(psi, phi_n, Nc, Ec, Vt)
    p = cc.p(psi, phi_p, Nv, Ev, Vt)
    Rdot = B*p*ndot
    return Rdot

def rad_dR_dphip(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, B):
    """
    Radiative recombination rate derivative with respect to hole quasi Fermi
    potential.
    """
    n = cc.n(psi, phi_n, Nc, Ec, Vt)
    pdot = cc.dp_dphip(psi, phi_p, Nv, Ev, Vt)
    Rdot = B*n*pdot
    return Rdot

def auger_R(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, Cn, Cp):
    """
    Auger recombination rate.
    """
    n = cc.n(psi, phi_n, Nc, Ec, Vt)
    p = cc.p(psi, phi_p, Nv, Ev, Vt)
    R = (Cn*n+Cp*p) * (n*p-n0*p0)
    return R

def auger_dR_dpsi(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, Cn, Cp):
    """
    Auger recombination rate derivative with respect to potential.
    """
    n = cc.n(psi, phi_n, Nc, Ec, Vt)
    ndot = cc.dn_dpsi(psi, phi_n, Nc, Ec, Vt)
    p = cc.p(psi, phi_p, Nv, Ev, Vt)
    pdot = cc.dp_dpsi(psi, phi_p, Nv, Ev, Vt)
    delta2 = 2*n*p - n0*p0
    Rdot = Cn*(delta2*ndot + n**2*pdot) + Cp*(delta2*pdot + p**2*ndot)
    return Rdot

def auger_dR_dphin(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, Cn, Cp):
    """
    Auger recombination rate derivative with respect to electron quasi Fermi
    potential.
    """
    n = cc.n(psi, phi_n, Nc, Ec, Vt)
    ndot = cc.dn_dphin(psi, phi_n, Nc, Ec, Vt)
    p = cc.p(psi, phi_p, Nv, Ev, Vt)
    Rdot = ndot * (Cn*(2*n*p - n0*p0) + Cp*p**2)
    return Rdot

def auger_dR_dphip(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, Cn, Cp):
    """
    Auger recombination rate derivative with respect to electron quasi Fermi
    potential.
    """
    n = cc.n(psi, phi_n, Nc, Ec, Vt)
    p = cc.p(psi, phi_p, Nv, Ev, Vt)
    pdot = cc.dp_dphip(psi, phi_p, Nv, Ev, Vt)
    Rdot = pdot * (Cn*n**2 + Cp*(2*n*p - n0*p0))
    return Rdot

# testing
test_err = 1e-3

def test_srh_derivatives():
    psi = 24
    phi_n = 20
    phi_p = 19
    Nc = 2.4e10
    Nv = 45e10
    Ec = 55
    Ev = -1
    Vt = 1.4
    ni = 2e7
    n0 = 2e7
    p0 = 1e7
    tau_n = 2
    tau_p = 3

    # psi
    psi_1 = psi*0.999
    psi_2 = psi*1.001
    R1 = srh_R(psi_1, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, ni, n0, p0,
               tau_n, tau_p)
    R2 = srh_R(psi_2, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, ni, n0, p0,
               tau_n, tau_p)
    Rdot_fd = (R2-R1)/(psi_2-psi_1)
    Rdot = srh_dR_dpsi(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, ni, n0, p0,
                       tau_n, tau_p)
    err_psi = abs(1-Rdot/Rdot_fd)

    # phi_n
    phi_n1 = phi_n*0.999
    phi_n2 = phi_n*1.001
    R1 = srh_R(psi, phi_n1, phi_p, Nc, Nv, Ec, Ev, Vt, ni, n0, p0,
               tau_n, tau_p)
    R2 = srh_R(psi, phi_n2, phi_p, Nc, Nv, Ec, Ev, Vt, ni, n0, p0,
               tau_n, tau_p)
    Rdot_fd = (R2-R1)/(phi_n2-phi_n1)
    Rdot = srh_dR_dphin(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, ni, n0, p0,
                        tau_n, tau_p)
    err_phin = abs(1-Rdot/Rdot_fd)

    # phi_p
    phi_p1 = phi_p*0.999
    phi_p2 = phi_p*1.001
    R1 = srh_R(psi, phi_n, phi_p1, Nc, Nv, Ec, Ev, Vt, ni, n0, p0,
               tau_n, tau_p)
    R2 = srh_R(psi, phi_n, phi_p2, Nc, Nv, Ec, Ev, Vt, ni, n0, p0,
               tau_n, tau_p)
    Rdot_fd = (R2-R1)/(phi_p2-phi_p1)
    Rdot = srh_dR_dphip(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, ni, n0, p0,
                        tau_n, tau_p)
    err_phip = abs(1-Rdot/Rdot_fd)

    # checking results
    assert err_psi < test_err and err_phin < test_err and err_phip < test_err

def test_rad_derivatives():
    psi = 24
    phi_n = 20
    phi_p = 19
    Nc = 2.4e10
    Nv = 45e10
    Ec = 55
    Ev = -1
    Vt = 1.4
    n0 = 1e6
    p0 = 4e6
    B = 1e-10

    # psi
    psi_1 = psi*0.999
    psi_2 = psi*1.001
    R1 = rad_R(psi_1, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, B)
    R2 = rad_R(psi_2, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, B)
    Rdot_fd = (R2-R1)/(psi_2-psi_1)
    Rdot = rad_dR_dpsi(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, B)
    err_psi = abs(1-Rdot/Rdot_fd)

    # phi_n
    phi_n1 = phi_n*0.999
    phi_n2 = phi_n*1.001
    R1 = rad_R(psi, phi_n1, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, B)
    R2 = rad_R(psi, phi_n2, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, B)
    Rdot_fd = (R2-R1)/(phi_n2-phi_n1)
    Rdot = rad_dR_dphin(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, B)
    err_phin = abs(1-Rdot/Rdot_fd)

    # phi_p
    phi_p1 = phi_p*0.999
    phi_p2 = phi_p*1.001
    R1 = rad_R(psi, phi_n, phi_p1, Nc, Nv, Ec, Ev, Vt, n0, p0, B)
    R2 = rad_R(psi, phi_n, phi_p2, Nc, Nv, Ec, Ev, Vt, n0, p0, B)
    Rdot_fd = (R2-R1)/(phi_p2-phi_p1)
    Rdot = rad_dR_dphip(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, B)
    err_phip = abs(1-Rdot/Rdot_fd)

    # checking results
    assert err_psi < test_err and err_phin < test_err and err_phip < test_err

def test_auger_derivatives():
    psi = 24
    phi_n = 20
    phi_p = 19
    Nc = 2.4e10
    Nv = 45e10
    Ec = 55
    Ev = -1
    Vt = 1.2
    n0 = 1e6
    p0 = 4e6
    Cn = 2e-30
    Cp = 3e-30

    # psi
    psi_1 = psi*0.999
    psi_2 = psi*1.001
    R1 = auger_R(psi_1, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, Cn, Cp)
    R2 = auger_R(psi_2, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, Cn, Cp)
    Rdot_fd = (R2-R1)/(psi_2-psi_1)
    Rdot = auger_dR_dpsi(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, Cn, Cp)
    err_psi = abs(1-Rdot/Rdot_fd)

    # phi_n
    phi_n1 = phi_n*0.999
    phi_n2 = phi_n*1.001
    R1 = auger_R(psi, phi_n1, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, Cn, Cp)
    R2 = auger_R(psi, phi_n2, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, Cn, Cp)
    Rdot_fd = (R2-R1)/(phi_n2-phi_n1)
    Rdot = auger_dR_dphin(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, Cn, Cp)
    err_phin = abs(1-Rdot/Rdot_fd)

    # phi_p
    phi_p1 = phi_p*0.999
    phi_p2 = phi_p*1.001
    R1 = auger_R(psi, phi_n, phi_p1, Nc, Nv, Ec, Ev, Vt, n0, p0, Cn, Cp)
    R2 = auger_R(psi, phi_n, phi_p2, Nc, Nv, Ec, Ev, Vt, n0, p0, Cn, Cp)
    Rdot_fd = (R2-R1)/(phi_p2-phi_p1)
    Rdot = auger_dR_dphip(psi, phi_n, phi_p, Nc, Nv, Ec, Ev, Vt, n0, p0, Cn, Cp)
    err_phip = abs(1-Rdot/Rdot_fd)

    # checking results
    assert err_psi < test_err and err_phin < test_err and err_phip < test_err