# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:24:15 2020

@author: vsgolovin
"""

import numpy as np

necessary = ['Ev', 'Ec', 'Nd', 'Na', 'Nc', 'Nv', 'mu_n' 'mu_p', 'tau_n',
             'tau_p' 'B', 'Cn', 'Cp', 'eps']

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

    def get_value(self, x):
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
        dx : number
            Layer thickness (cm).
        """
        assert isinstance(name, str)
        self.name = name
        assert isinstance(dx, (float, int))
        self.dx = float(dx)

        # storing necessary (n) parameters` names in a list
        # and creating a dictionary for storing PhysParam objects
        self.params = dict()  # dictionary of parameters

    # getters
    def get_name(self):
        """
        Return layer name.
        """
        return self.name

    def get_thickness(self):
        """
        Return layer thickness (cm).
        """
        return self.dx

    # setters for physical parameters
    def set_parameter(self, p, y1=None, y2=None, y_fun=None):
        """
        Add a `PhysParam` object to the layer.

        Parameters
        ----------
        p : str
            Parameter name.
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
        param = PhysParam(name=p, dx=self.dx, y1=y1, y2=y2, y_fun=y_fun)
        self.params[p] = param

    def check_parameters(self, nparams):
        """
        Check if all the necessary input parameters were specified.

        Parameters
        ----------
        nparams : iterable
            List or tuple of necessary physical parameters (str).

        Returns
        -------
        success : bool
            `True` if every parameter was specified, `False` otherwise.
        missing : list
            List of missing parameters' names.
        """
        success = True
        missing = list()
        for s in nparams:
            if s not in self.params.keys():
                success = False
                missing.append(s)
        return success, missing

    def prepare(self, nparams):
        """
        Make sure all the needed parameters are specified or calculated.

        Parameters
        ----------
        nparams : iterable
            List or tuple of necessary physical parameters (str).

        Raises
        ------
        Exception
            If some necessary input parameters were not specified.
        """
        # checking necessary physical parameters
        success, missing = self.check_parameters(nparams)
        if not success:
            msg_1 = 'Layer '+self.name+': '
            msg_3 = 'not specified'
            if len(missing)==1:
                msg_2 = 'parameter '+str(missing[0])+' was '
            else:
                msg_2 = 'parameters '+', '.join(missing[:-1])+\
                        ' and '+str(missing[-1])+' were '
            msg = msg_1+msg_2+msg_3
            raise Exception(msg)

        # calculating some useful parameters
        # doping profile C_dop = Nd - Na
        if ('Nd' in self.params) and ('Na' in self.params):
            f = lambda x: ( self.params['Nd'].get_value(x)
                           -self.params['Na'].get_value(x))
            self.set_parameter('C_dop', y_fun=f)

        # bandgap Eg = Ec - Ev
        if ('Ec' in self.params) and ('Ev' in self.params):
            f = lambda x: ( self.params['Ec'].get_value(x)
                           -self.params['Ev'].get_value(x))
            self.set_parameter('Eg', y1=None, y2=None, y_fun=f)

    def get_value(self, p, x):
        """
        Get value of the physical parameter `p` at location `x`.

        Parameters
        ----------
        p : str
            Physical parameter name (i.e. 'Ec`).
        x : number
            x coordinate inside a layer.
        """
        pv = self.params[p]  # raises KeyError if p is not in keys
        y = pv.get_value(x)
        return y

class Device(object):

    def __init__(self):
        """
        Class for storing a collection of `Layer` objects, each corresponding
        to a particular index.
        """
        self.layers = dict()
        self.ind_max = -1
        self.ready = False

    def add_layer(self, l, ind=-1):
        """
        Add a layer to device.

        Parameters
        ----------
        l : Layer
            Layer to be added.
        ind : int, optional
            Index describing layer location in the device. Smaller indices
            correspond to smaller x coordinates. Default value is `-1`, that is
            the new layer is added to the top of the device (largest index and
                                                             largest x).
        """
        assert isinstance(ind, int) and (ind>=0 or ind==-1)
        if ind==-1:
            self.layers[self.ind_max+1] = l
            self.ind_max += 1
        else:
            self.layer[ind] = l
            if ind>self.ind_max:
                self.ind_max = ind
        self.ready = False

    def prepare(self):
        """
        Prepare object for calculations:
        1. Check if all the necessary parameters are specified in every layer.
        2. Create a list of sorted layer indices `inds`.
        3. Create a `numpy.ndarray` of layer boundaries `x_b`.
        """
        self.inds = sorted(self.layers.keys())
        self.x_b = np.zeros(len(self.inds)+1)
        i = 1
        for ind in self.inds:
            l = self.layers[ind]
            l.prepare()
            dx = l.get_thickness()
            self.x_b[i] = self.x_b[i-1] + dx
            i += 1
        self.ready = True

    def get_value(self, p, x):
        """
        Calculate parameter's `p` value at location `x`.

        Parameters
        ----------
        p : str
            Physical parameter name (i.e. 'Ec`).
        x : number
            x coordinate inside a device.
        """
        # checking arguments
        assert isinstance(p, str)
        assert x <= self.x_b[-1]

        # checking object
        if not self.ready:
            self.prepare()

        # finding layer
        for i in range(len(self.inds)):
            if x <= self.x_b[i+1]:
                break
        x_rel = x - self.x_b[i]
        ind = self.inds[i]
        l = self.layers[ind]

        # calculating value
        y = l.get_value(p, x_rel)
        return y

# some unnecessary tests
def test_physparam_y1():
    """Only y1 is specified."""
    p = PhysParam("mu_n", 0.1, y1=300)
    y = p.get_value(0.07)
    eps = 1e-6
    success = np.abs(y-300)<eps
    assert success

def test_physparam_y1y2():
    """Both y1 and y2 are specified."""
    p = PhysParam("Ec", 10, y1=1, y2=-1)
    x = np.array([0, 2, 5, 10])
    y = p.get_value(x)
    y_correct = np.array([1, 0.6, 0.0, -1.0])
    eps = 1e-6
    dy = np.abs(y-y_correct)
    success = (dy<eps).all()
    assert success

def test_physparam_yfun():
    """y_fun is used for calculation instead of y1 and y2."""
    p = PhysParam("Ec", 10, y1=1, y2=-1, y_fun=lambda x: x**2)
    x = np.array([0, 2, 5, 10])
    y = p.get_value(x)
    y_correct = np.array([0, 4, 25, 100])
    eps = 1e-6
    dy = np.abs(y-y_correct)
    success = (dy<eps).all()
    assert success

def test_layer_cs():
    """All the needed parameters were specified."""
    l = Layer('nclad', 1.5e-4)
    for p in necessary:
        l.set_parameter(p, 1.0)
    success, _ = l.check_parameters(necessary)
    assert success

def test_layer_cf():
    """Unspecified necessary parameter is correctly identified."""
    inp_params = necessary.copy()
    inp_params.remove('Na')
    l = Layer('nclad', 1.0e-4)
    for p in inp_params:
        l.set_parameter(p, 1.0)
    s, m = l.check_parameters(necessary)
    success = (not s) and ('Na' in m) and (len(m)==1)
    assert success

def test_layer_Eg():
    """Layer.prepare() executes and Eg is correctly defined."""
    n_params = ['Ec', 'Ev', 'Nd']
    l = Layer('nclad', 1.5e-4)
    l.set_parameter('Ec', 1.5)
    l.set_parameter('Ev', 0)
    l.set_parameter('Nd', 1e18, 1e16)
    l.prepare(n_params)
    Eg_calc = l.get_value('Eg', 0.5e-4)
    Eg_real = 1.5
    eps = 1e-6
    success = (np.abs(Eg_calc-Eg_real) < eps)
    assert success
