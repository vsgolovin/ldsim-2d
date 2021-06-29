# -*- coding: utf-8 -*-
"""
Classes for defining laser diode vertical (epitaxial) and lateral design.
"""

import numpy as np


params = ['Ev', 'Ec', 'Nd', 'Na', 'Nc', 'Nv', 'mu_n', 'mu_p', 'tau_n',
          'tau_p', 'B', 'Cn', 'Cp', 'eps', 'n_refr', 'Eg', 'C_dop',
          'fca_e', 'fca_h']
params_active = ['g0', 'N_tr']


class Layer(object):
    def __init__(self, name, dx, active=False):
        """
        Parameters:
            name : str
                Layer name.
            dx : float
                Layer thickness (cm).
            active : bool
                Whether layer is part of laser diode active region.
        """
        self.name = name
        self.dx = dx
        self.active = active
        self.d = dict.fromkeys(params, [np.nan])
        self.d['C_dop'] = self.d['Nd'] = self.d['Na'] = [0.0]
        if active:
            self.d.update(dict.fromkeys(params_active, [np.nan]))

    def __repr__(self):
        s1 = 'Layer \"{}\"'.format(self.name)
        s2 = '{} um'.format(self.dx * 1e4)
        Cdop = self.calculate('C_dop', [0, self.dx])
        if Cdop[0] > 0 and Cdop[1] > 0:
            s3 = 'n-type'
        elif Cdop[0] < 0 and Cdop[1] < 0:
            s3 = 'p-type'
        else:
            s3 = 'i-type'
        if self.active:
            s4 = 'active'
        else:
            s4 = 'not active'
        return ' / '.join((s1, s2, s3, s4))

    def __str__(self):
        return self.name

    def __eq__(self, other_name):
        assert isinstance(other_name, str)
        return self.name == other_name

    def calculate(self, param, x):
        "Calculate value of parameter `param` at location `x`."
        p = self.d[param]
        return np.polyval(p, x)

    def update(self, d):
        "Update polynomial coefficients of parameters."
        assert isinstance(d, dict)
        for k, v in d.items():
            if k not in self.d:
                raise Exception(f'Unknown parameter {k}')
            if isinstance(v, (int, float)):
                self.d[k] = [v]
            else:
                self.d[k] = v
        if 'Ec' in d or 'Ev' in d:
            self._update_Eg()
        if 'Nd' in d or 'Na' in d:
            self._update_Cdop()

    def _update_Eg(self):
        p_Ec = np.asarray(self.d['Ec'])
        p_Ev = np.asarray(self.d['Ev'])
        delta = len(p_Ec) - len(p_Ev)
        if delta > 0:
            p_Ev = np.concatenate([np.zeros(delta), p_Ev])
        elif delta < 0:
            p_Ec = np.concatenate([np.zeros(-delta), p_Ec])
        self.d['Eg'] = p_Ec - p_Ev

    def _update_Cdop(self):
        p_Nd = np.asarray(self.d['Nd'])
        p_Na = np.asarray(self.d['Na'])
        delta = len(p_Nd) - len(p_Na)
        if delta > 0:
            p_Na = np.concatenate([np.zeros(delta), p_Na])
        elif delta < 0:
            p_Nd = np.concatenate([np.zeros(-delta), p_Nd])
        self.d['C_dop'] = p_Nd - p_Na


class EpiDesign(list):
    "A list of `Layer` objects."
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def boundaries(self):
        "Get an array of layer boundaries."
        return np.cumsum([0.0] + [layer.dx for layer in self])

    def get_thickness(self):
        "Get sum of all layers' thicknesses."
        return self.boundaries()[-1]

    def _ind_dx(self, x):
        if x == 0:
            return 0, 0.0
        xi = 0.0
        for i, layer in enumerate(self):
            if x <= (xi + layer.dx):
                return i, x - xi
            xi += layer.dx
        return np.nan, np.nan

    def _inds_dx(self, x):
        inds = np.zeros_like(x)
        dx = np.zeros_like(x)
        for i, xi in enumerate(x):
            inds[i], dx[i] = self._ind_dx(xi)
        return inds, dx

    def calculate(self, param, x, inds=None, dx=None):
        "Calculate values of `param` at locations `x`."
        y = np.zeros_like(x)
        if isinstance(x, (float, int)):
            ind, dx = self._ind_dx(x)
            return self[ind].calculate(param, dx)
        else:
            if inds is None or dx is None:
                inds, dx = self._inds_dx(x)
            for i, layer in enumerate(self):
                ix = (inds == i)
                if ix.any():
                    y[ix] = layer.calculate(param, dx[ix])
        return y


class Design2D(object):
    "Vertical-lateral (x-y) laser diode design."
    def __init__(self, epi, width):
        """
        Parameters
        ----------
        epi : EpiDesign
            Epitaxial design.
        width : float
            Device total width (cm).
        """
        self.epi = epi
        self.ymax = width
        self.xmax = epi.get_thickness()
        # trench parameters
        self.dx = 0.0  # trench depth
        self.y1 = 0.0
        self.y2 = self.ymax / 3

    def inside(self, x, y):
        assert y <= self.ymax
        if y > self.ymax / 2:
            y = self.ymax - y
        if y < self.y1:
            return x <= (self.xmax - self.dx)
        elif y < self.y2:
            k = self.dx / (self.y2 - self.y1)
            return x <= ((self.xmax - self.dx) + (y - self.y1) * k)
        else:
            return x <= self.xmax

    def add_trenches(self, y1, y2, dx):
        """
        Add two trenches width depth `dx` to both sides of the device.
        """
        assert y1 < self.ymax / 2 and y2 < self.ymax / 2
        self.y1 = y1
        self.y2 = y2
        assert dx < self.xmax
        self.dx = dx


if __name__ == '__main__':

    d0 = dict(Ev=0.0, Ec=1.424, Nc=4.7e17, Nv=9.0e18, mu_n=8000,
              mu_p=370, tau_n=5e-9, tau_p=5e-9, B=1e-10, Cn=2e-30,
              Cp=2e-30, eps=12.9, n_refr=3.493,
              fca_e=4e-18, fca_h=12e-18)
    d25 = dict(Ev=-0.125, Ec=1.611, Nc=6.1e17, Nv=1.1e19, mu_n=3125,
               mu_p=174, tau_n=5e-9, tau_p=5e-9, B=1e-10, Cn=2e-30,
               Cp=2e-30, eps=12.19, n_refr=3.443,
               fca_e=4e-18, fca_h=12e-18)
    d40 = dict(Ev=-0.2, Ec=1.724, Nc=7.5e17, Nv=1.2e19, mu_n=800,
               mu_p=100, tau_n=5e-9, tau_p=5e-9, B=1e-10, Cn=2e-30,
               Cp=2e-30, eps=11.764, n_refr=3.351,
               fca_e=4e-18, fca_h=12e-18)

    ncl = Layer(name='n-cladding', dx=1.5e-4)
    ncl.update(d40)
    ncl.update({'Nd': 5e17})
    nwg = Layer(name='n-waveguide', dx=0.5e-4)
    nwg.update(d25)
    nwg.update({'Nd': 1e17})
    act = Layer(name='active', dx=100e-7, active=True)
    act.update(d0)
    act.update({'Nd': 2e16, 'g0': 1500, 'N_tr': 1.85e18})
    pwg = Layer(name='p-waveguide', dx=0.5e-4)
    pwg.update(d25)
    pwg.update({'Na': 1e17})
    pcl = Layer(name='p-cladding', dx=1.5e-4)
    pcl.update(d40)
    pcl.update({'Na': 1e18})

    pin = EpiDesign((ncl, nwg, act, pwg, pcl))
    cs = Design2D(pin, 130e-4)
    cs.add_trenches(y1=10e-4, y2=20e-4, dx=1.7e-4)

    x = np.linspace(0, pin.get_thickness(), 1000)
    y = np.linspace(0, cs.ymax, 121)

    Z = np.zeros((len(x), len(y)), dtype='bool')
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            Z[i, j] = cs.inside(xi, yj)

    Eg = pin.calculate('Eg', x)
    Eg_2D = np.repeat(Eg, len(y)).reshape(len(x), len(y))
    Eg_2D[Z == False] = 0.0

    import matplotlib.pyplot as plt
    plt.close('all')
    plt.figure()
    plt.contourf(y, x, Eg_2D, cmap=plt.cm.Blues)
    plt.show()
