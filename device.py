# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:24:15 2020

@author: vsgolovin
"""

import numpy as np

class PhysParam(object):
    def __init__(self, name, dx, y1=None, y2=None, y_fun=None):
        """
        A class for storing values of physical parameters (mobility, doping
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
        assert isinstance(dx, float) or isinstance(dx, int)
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
            assert isinstance(y1, float) or isinstance(y1, int)
            if y2 is not None:
                assert isinstance(y2, float) or isinstance(y2, int)
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
            assert isinstance(x, float) or isinstance(x, int)
            assert x>=0 and x<=self.dx
        return self.y_fun(x)

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