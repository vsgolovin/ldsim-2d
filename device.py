# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:24:15 2020

@author: vsgolovin
"""

import numpy as np

class PhysParam(object):

    def __init__(self, name, dx, y1=None, y2=None, y_fun=None):
        """
        Class for storing values of physical parameters (mobility, doping
        density, etc) in a layer of thickness `dx`.

        Parameters
        ----------
        name : str
            Name of the physical parameter.
        dx : number
            Thickness of the corresponding layer.
        y1 : number or NoneType, Optional
            Value of the physical parameter at x=0. Not needed if `y_fun` is
            specified.
        y2 : number or NoneType, Optional
            Value of the physical parameter at x=`dx`. Values between 0 and
            'dx' are calculated using linear interpolation. Not needed if
            `y_fun` is specified.
        y_fun : function or NoneType, Optional
            Function for calculating physical parameter value at arbitrary x
            between 0 and `dx`. Should have only one argument. Takes precedence
            over `y1` and `y2`.
        """
        assert isinstance(dx, (float, int))
        self.dx = float(dx)
        assert isinstance(name, str)
        self.name = name

        # picking a function for calculating y
        if y_fun is not None:  # function is passed as y_fun
            try:  # simple tests of y_fun
                y_fun(0)
                y_fun(self.dx/2)
                y_fun(self.dx)
            except:
                raise Exception('Function y_fun is incorrectly defined.')
            self.y_fun = y_fun
        else:  # y_fun is None
            assert isinstance(y1, (float, int))
            if y2 is not None:
                assert isinstance(y2, (float, int))
                k = (y2-y1) / self.dx
                self.y_fun = lambda x: y1+k*x
            else:
                self.y_fun = lambda x: y1

    def get_name(self):
        """
        Return parameter name (str)
        """
        return self.name

    def value(self, x):
        """
        Return physical parameter value at x=`x`.
        """
        if isinstance(x, np.ndarray):
            assert x.max()<=self.dx and x.min()>=0
        else:
            assert isinstance(x, (float, int))
            assert x>=0 and x<=self.dx
        return self.y_fun(x)

class Layer(object):

    def __init__(self, name, dx):
        """
        Class for storing a collection of `PhysParam` objects.

        Parameters
        ----------
        name : str
            Layer name.
        """
        assert isinstance(name, str)
        self.name = name
        assert isinstance(dx, (float, int))
        self.dx = float(dx)

        # storing necessary (n) and optional (o) parameters` names in lists
        # and PhysParam objects in a dictionary
        self.n_params = ['Ev', 'Ec', 'Nd', 'Na', 'Nc', 'Nv', 'mu_n',
                             'mu_p', 'tau_n', 'B', 'Cn']
        self.o_params = ['tau_p', 'Et', 'Cp']
        self.params = dict()  # dictionary of parameters

    def check_n_params(self):
        """
        Check if all the necessary input parameters were specified.

        Returns
        -------
        success : bool
            `True` if every parameter was specified, `False` otherwise.
        missing : list
            List of missing parameters' names.
        """
        success = True
        missing = list()
        for s in self.n_params:
            if s not in self.params.keys():
                success = False
                missing.append(s)
        return success, missing

    # setters for physical parameters
    def _set_param(self, s, y1, y2, y_fun):
        """
        Method for adding a generic `PhysParam` object to `self.params`
        dictionary.
        """
        param = PhysParam(name=s, dx=self.dx, y1=y1, y2=y2, y_fun=y_fun)
        self.params[s] = param

    def set_Ev(self, y1=None, y2=None, y_fun=None):
        """
        Set the valence band edge energy (eV).
        """
        self._set_param('Ev', y1, y2, y_fun)

    def set_Ec(self, y1=None, y2=None, y_fun=None):
        """
        Set the conduction band edge energy (eV).
        """
        self._set_param('Ec', y1, y2, y_fun)

    def set_Nd(self, y1=None, y2=None, y_fun=None):
        """
        Set the donor doping concentration (cm-3).
        """
        self._set_param('Nd', y1, y2, y_fun)

    def set_Na(self, y1=None, y2=None, y_fun=None):
        """
        Set the acception doping concentration (cm-3).
        """
        self._set_param('Na', y1, y2, y_fun)

    def set_Nc(self, y1=None, y2=None, y_fun=None):
        """
        Set the effective density of states in the conduction band (cm-3).
        """
        self._set_param('Nc', y1, y2, y_fun)

    def set_Nv(self, y1=None, y2=None, y_fun=None):
        """
        Set the effective density of states in the valence band (cm-3).
        """
        self._set_param('Nv', y1, y2, y_fun)

    def set_mu_n(self, y1=None, y2=None, y_fun=None):
        """
        Set the electron mobility (cm2 V-1 s-1).
        """
        self._set_param('mu_n', y1, y2, y_fun)

    def set_mu_p(self, y1=None, y2=None, y_fun=None):
        """
        Set the hole mobility (cm2 V-1 s-1).
        """
        self._set_param('mu_p', y1, y2, y_fun)

    def set_tau_n(self, y1=None, y2=None, y_fun=None):
        """
        Set the electron lifetime due to Shockley-Read-Hall recombination (s).
        """
        self._set_param('tau_n', y1, y2, y_fun)
        if 'tau_p' not in self.params:
            self._set_param('tau_p', y1, y2, y_fun)

    def set_tau_p(self, y1=None, y2=None, y_fun=None):
        """
        Set the hole lifetime due to Shockley-Read-Hall recombination (s).
        """
        self._set_param('tau_p', y1, y2, y_fun)

    def set_Et(self, y1=None, y2=None, y_fun=None):
        """
        Set the energy level of deep traps (eV).
        """
        self._set_param('Et', y1, y2, y_fun)

    def set_B(self, y1=None, y2=None, y_fun=None):
        """
        Set the radiative recombination coefficient (cm3 s-1).
        """
        self._set_param('B', y1, y2, y_fun)

    def set_Cn(self, y1=None, y2=None, y_fun=None):
        """
        Set the electron Auger recombination coefficient (cm6 s-1).
        """
        self._set_param('Cn', y1, y2, y_fun)
        if 'Cp' not in self.params:
            self._set_param('Cp', y1, y2, y_fun)

    def set_Cp(self, y1=None, y2=None, y_fun=None):
        """
        Set the hole Auger recombination coefficient (cm6 s-1).
        """
        self._set_param('Cp', y1, y2, y_fun)

# some unnecessary tests
def test_physparam_y1():
    """Only y1 is specified."""
    p = PhysParam("mu_n", 0.1, y1=300)
    y = p.value(0.07)
    eps = 1e-6
    success = np.abs(y-300)<eps
    assert success

def test_physparam_y1y2():
    """Both y1 and y2 are specified."""
    p = PhysParam("Ec", 10, y1=1, y2=-1)
    x = np.array([0, 2, 5, 10])
    y = p.value(x)
    y_correct = np.array([1, 0.6, 0.0, -1.0])
    eps = 1e-6
    dy = np.abs(y-y_correct)
    success = (dy<eps).all()
    assert success

def test_physparam_yfun():
    """y_fun is used for calculation instead of y1 and y2."""
    p = PhysParam("Ec", 10, y1=1, y2=-1, y_fun=lambda x: x**2)
    x = np.array([0, 2, 5, 10])
    y = p.value(x)
    y_correct = np.array([0, 4, 25, 100])
    eps = 1e-6
    dy = np.abs(y-y_correct)
    success = (dy<eps).all()
    assert success
