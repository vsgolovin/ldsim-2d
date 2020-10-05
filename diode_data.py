#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:15:34 2020

@author: vsgolovin
"""

import numpy as np
from device import Device

# DiodeData-specific necessary input parameters
input_params = ['Ev', 'Ec', 'Nd', 'Na', 'Nc', 'Nv', 'mu_n' 'mu_p', 'tau_n',
                'tau_p' 'B', 'Cn', 'Cp', 'eps']

class DiodeData(object):

    def _generate_uniform_mesh(self, device, step):
        """
        Generates a uniform mesh. Step may be adjusted if it does not divide
        device thickness.
        """
        width = device.get_thickness()
        assert isinstance(step, (int, float)) and step < width/2
        n = int(np.round(width/step, 0))
        self.x = np.linspace(0, width, n)

    def __init__(self, device, mesh='uniform', **kwargs):
        """
        """
        params = device.get_params()
        assert all([p in params for p in input_params])
        assert device.ready

        # creating 1D mesh
        if mesh=='uniform':
            assert 'step' in kwargs
            self._generate_uniform_mesh(device, kwargs['step'])
        else:
            raise NotImplementedError

        # calculating physical parameters' values at mesh nodes
        pass