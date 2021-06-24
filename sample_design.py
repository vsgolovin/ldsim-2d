# -*- coding: utf-8 -*-
"""
Sample epitaxial design.
"""

from design import Layer, EpiDesign


# material parameters for different AlGaAs alloys
d0 = dict(Ev=0.0, Ec=1.424, Nc=4.7e17, Nv=9.0e18, mu_n=8000, mu_p=370,
          tau_n=5e-9, tau_p=5e-9, B=1e-10, Cn=2e-30, Cp=2e-30,
          eps=12.9, n_refr=3.493, fca_e=4e-18, fca_h=12e-18)
d25 = dict(Ev=-0.125, Ec=1.611, Nc=6.1e17, Nv=1.1e19, mu_n=3125, mu_p=174,
           tau_n=5e-9, tau_p=5e-9, B=1e-10, Cn=2e-30, Cp=2e-30,
           eps=12.19, n_refr=3.443, fca_e=4e-18, fca_h=12e-18)
d40 = dict(Ev=-0.2, Ec=1.724, Nc=7.5e17, Nv=1.2e19, mu_n=800, mu_p=100,
           tau_n=5e-9, tau_p=5e-9, B=1e-10, Cn=2e-30, Cp=2e-30,
           eps=11.764, n_refr=3.351, fca_e=4e-18, fca_h=12e-18)


# create Layer objects
ncont = Layer(name='n-contact', dx=0.3e-4)
ncont.update(d0)
ncont.update({'Nd': 1e18})

ncl = Layer(name='n-cladding', dx=1.5e-4)
ncl.update(d40)
ncl.update({'Nd': 5e17})

nwg = Layer(name='n-waveguide', dx=0.5e-4)
nwg.update(d25)
nwg.update({'Nd': 1e17})

act = Layer(name='active', dx=0.03e-4, active=True)
act.update(d0)
act.update({'Nd': 2e16, 'g0': 1500, 'N_tr': 1.85e18})

pwg = Layer(name='p-waveguide', dx=0.5e-4)
pwg.update(d25)
pwg.update({'Na': 1e17})

pcl = Layer(name='p-cladding', dx=1.5e-4)
pcl.update(d40)
pcl.update({'Na': 1e18})

pcont = Layer(name='p-contact', dx=0.3e-4)
pcont.update(d0)
pcont.update({'Na': 2e18})


# create design as a list of layers
epi = EpiDesign([ncont, ncl, nwg, act, pwg, pcl, pcont])
