# -*- coding: utf-8 -*-
"""
Tools for setting up the problem. Class `Slice` is used to define values of
all the input parameters (doping densities, energy band boundaries, carrier
mobilities, etc.) for every x in a 1D slice of a device.
"""

import numpy as np

class PhysParam(object):

    def __init__(self, name, dx, y):
        """
        Class for storing values of physical parameters (mobility, doping
        density, etc) in a layer of thickness `dx`.

        Parameters
        ----------
        name : str
            Name of the physical parameter.
        dx : number
            Thickness of the corresponding layer.
        y : number, tuple or function
            Parameter value. Possible formats:
                * Single number. In this case parameter value does not depend
                  on x coordinate.
                * Two numbers (tuple or list). Here parameter value is a linear
                  function of x, y(x=0) = `y[0]`, y(x=dx) = `y[1]`.
                * Function of x. Should be defined at [0, `dx`].
        """
        assert isinstance(dx, (float, int))
        self.dx = float(dx)
        assert isinstance(name, str)
        self.name = name

        # picking a function for calculating y
        if callable(y):
            try:  # simple tests
                y(0)
                y(self.dx/2)
                y(self.dx)
            except:
                raise Exception('Function y is incorrectly defined.')
            self.y_fun = y
        elif isinstance(y, (tuple, list)):
            assert len(y)==2 and all([isinstance(yi, (float, int)) for yi in y])
            self.y_fun = lambda x: y[0] + ((y[1]-y[0])/dx) * x
        elif isinstance(y, (int, float)):
            self.y_fun = lambda x: y
        else:
            raise Exception('Parameter %s: y is incorrectly defined' % name)

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
            assert isinstance(x, (float, int)) and x>=0 and x<=self.dx
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

        # creating a dictionary for storing PhysParam objects
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
    def set_parameter(self, p, y):
        """
        Add a `PhysParam` object to the layer.

        Parameters
        ----------
        p : str
            Parameter name.
        y : number, tuple or function
            Parameter value. Possible formats:
                * Single number. In this case parameter value does not depend
                  on x coordinate.
                * Two numbers (tuple or list). Here parameter value is a linear
                  function of x, y(x=0) = `y[0]`, y(x=dx) = `y[1]`.
                * Function of x. Should be defined at [0, `dx`].
        """
        param = PhysParam(name=p, dx=self.dx, y=y)
        self.params[p] = param

    def check_parameters(self, nparams):
        """
        Check if all the necessary input parameters were specified.

        Parameters
        ----------
        nparams : iterable
            List or tuple of necessary physical parameters' names.

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

    def check(self, nparams):
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

class Slice(object):

    def __init__(self):
        """
        Class for storing a collection of `Layer` objects, each corresponding
        to a particular index.
        """
        self.layers = dict()
        self.ind_max = -1       # top layer index
        self.inds = []          # sorted keys of `self.layers`
        self.x_b = np.empty(0)  # layer boundaries' coordinates

    def _update(self):
        """
        Update:
        1. List of sorted layer indices `inds`.
        2. `numpy.ndarray` of layer boundaries `x_b`.
        """
        self.inds = sorted(self.layers.keys())
        self.x_b = np.zeros(len(self.inds)+1)
        i = 1
        for ind in self.inds:
            l = self.layers[ind]
            dx = l.get_thickness()
            self.x_b[i] = self.x_b[i-1] + dx
            i += 1

    def add_layer(self, l, ind=-1):
        """
        Add a layer to slice.

        Parameters
        ----------
        l : Layer
            Layer to be added.
        ind : int, optional
            Index describing layer location in the slice. Smaller indices
            correspond to smaller x coordinates. Default value is `-1`, that is
            the new layer is added to the top of the slice (largest index and
                                                            largest x).
        """
        assert isinstance(ind, int) and (ind>=0 or ind==-1)
        if ind==-1:
            self.layers[self.ind_max+1] = l
            self.ind_max += 1
        else:
            self.layers[ind] = l
            if ind>self.ind_max:
                self.ind_max = ind
        self._update()  # updating arrays of indices and boundaries

    def get_thickness(self):
        """
        Get total slice thickness.
        """
        return self.x_b[-1]

    def get_index(self, x, get_xrel=False):
        """
        For a given coordinate `x` return index of corresponding layer.
        Optionally returns location inside the layer.

        Parameters
        ----------
        x : number
            x coordinate inside slice.
        get_xrel : bool, optional
            Whether to return relative position inside layer.

        """
        b = self.x_b  # layer boundaries
        assert x>=b[0] and x<=b[-1]
        b = b[:-1]
        n = len(b)
        ix = 0
        # bijection search
        while n>1:
            n = n//2
            if x<b[n]:
                b = b[:n]
            else:
                ix += n
                b = b[n:]
                n = len(b)

        ind = self.inds[ix]
        if get_xrel:
            x_rel = x - self.x_b[ix]
            return ind, x_rel
        return ind

    def get_value(self, p, x):
        """
        Calculate parameter's `p` value at location `x`.

        Parameters
        ----------
        p : str
            Physical parameter name (i.e. 'Ec`).
        x : number
            x coordinate inside a slice.
        """
        # checking parameter name
        assert isinstance(p, str)

        # finding layer
        ind, x_rel = self.get_index(x, get_xrel=True)
        l = self.layers[ind]

        # calculating value
        y = l.get_value(p, x_rel)
        return y

# some unnecessary tests
def test_physparam_num():
    "Single number as y."
    p = PhysParam("mu_n", 0.1, y=300)
    y = p.get_value(0.07)
    eps = 1e-6
    success = np.abs(y-300)<eps
    assert success

def test_physparam_dual():
    "y is two numbers."
    p = PhysParam("Ec", 10, [1, -1])
    x = np.array([0, 2, 5, 10])
    y = p.get_value(x)
    y_correct = np.array([1, 0.6, 0.0, -1.0])
    eps = 1e-6
    dy = np.abs(y-y_correct)
    success = (dy<eps).all()
    assert success

def test_physparam_fun():
    "y is a function."
    p = PhysParam("Ec", 10, y=lambda x: x**2)
    x = np.array([0, 2, 5, 10])
    y = p.get_value(x)
    y_correct = np.array([0, 4, 25, 100])
    eps = 1e-6
    dy = np.abs(y-y_correct)
    success = (dy<eps).all()
    assert success

def test_layer_cs():
    """All the needed parameters were specified."""
    necessary = ['Ec', 'Ev', 'Nc', 'Nv', 'foo', 'bar']
    l = Layer('nclad', 1.5e-4)
    for p in necessary:
        l.set_parameter(p, 1.0)
    success, _ = l.check_parameters(necessary)
    assert success

def test_layer_cf():
    """Unspecified necessary parameter is correctly identified."""
    necessary = ['Ec', 'Ev', 'Nc', 'Nv', 'foo', 'bar']
    inp_params = necessary.copy()
    inp_params.remove('foo')
    l = Layer('nclad', 1.0e-4)
    for p in inp_params:
        l.set_parameter(p, 1.0)
    s, m = l.check_parameters(necessary)
    success = (not s) and ('foo' in m) and (len(m)==1)
    assert success

def test_slice():
    """
    Testing if doping profile is correctly evaluated in
    a two-layer slice.
    """
    # creating layers
    l1 = Layer('n', 1e-4)
    l1.set_parameter('Nd', 1e18)
    l1.set_parameter('Na', 2e17)
    l2 = Layer('p', 2e-4)
    l2.set_parameter('Nd', 4e17)
    l2.set_parameter('Na', 9e17)

    # assembling slice
    d = Slice()
    d.add_layer(l2, 1)
    d.add_layer(l1, 0)

    # calculating and checking doping profile
    x = np.array([0, 0.5e-4, 0.99e-4, 1.01e-4, 2e-4, 3e-4])
    Cdop_real = np.array([1e18-2e17]*3+[4e17-9e17]*3)
    Nd_calc = np.array([d.get_value('Nd', xi) for xi in x])
    Na_calc = np.array([d.get_value('Na', xi) for xi in x])
    Cdop_calc = Nd_calc - Na_calc
    err = np.abs((Cdop_calc-Cdop_real)/Cdop_real)
    success = (err<1e-6).all()
    assert success

def test_slice_thickness():
    """
    Testing the Slice.get_thickness() method.
    """
    # creating layers
    l1 = Layer('AlGaAs', 1.5e-4)
    l1.set_parameter('Ev', 0)
    l1.set_parameter('Ec', 1.8)
    l2 = Layer('GaAs', 1e-4)
    l2.set_parameter('Ev', 0.2)
    l2.set_parameter('Ec', 1.6)

    # assembling device and calculating for both cases
    d = Slice()
    d.add_layer(l1)
    d.add_layer(l2)
    x1 = d.get_thickness()
    x2 = d.get_thickness()

    # checking values
    eps = 1e-6
    x_real = 1.5e-4+1e-4
    eq1 = np.abs((x1-x_real)/x_real) < eps
    eq2 = np.abs((x2-x_real)/x_real) < eps
    success = eq1 and eq2
    assert success
