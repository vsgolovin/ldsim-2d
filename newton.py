# -*- coding: utf-8 -*-
"""
Defines a class for solving systems of equations using Newton's method.
"""

import numpy as np

def l2_norm(x):
    "Calculate L2 (Euclidean) norm of vector `x`."
    x = np.asarray(x)
    return np.sqrt(np.sum(x*x))

class NewtonSolver(object):

    def __init__(self, res, jac, x0, linalg_solver):
        """
        Class for solving a system of equations using Newton's method.

        Parameters
        ----------
        res : callable
            Right-hand side (residual) of the system.
        jac : callable
            Jacobian of the system.
        x0 : array-like
            Initial guess.
        linalg_solver : callable
            Method for solving the 'A*x = b` system.

        """
        self.rfun = res
        self.jfun = jac
        self.x = x0
        self.la_solver = linalg_solver
        self.i = 0  # iteration number
        self.rnorms = list()  # L2 norms of residuals
        self.fluct = list()   # fluctuation values -- ||dx|| / ||x||

    def step(self, omega=1.0):
        """
        Perform single iteration and update x.

        Parameters
        ----------
        omega : float
            Damping parameter, 0<`omega`<=1.

        """
        # check omega value
        assert omega>0 and omega<=1.0

        # calculate residual and Jacobian
        self.rvec = self.rfun(self.x)
        self.rnorms.append(l2_norm(self.rvec))
        self.jac = self.jfun(self.x)

        # solve J*dx = -r system and update x
        dx = self.la_solver(self.jac, -self.rvec)
        self.fluct.append(l2_norm(dx)/l2_norm(self.x))
        self.x += dx*omega
        self.i += 1

if __name__=='__main__':
    # solving a simple nonlinear system
    import matplotlib.pyplot as plt
     
    def residual(x):
         r = np.empty(2)
         r[0] = 2*x[0]**2 + 3*x[1] - 8
         r[1] = 3*x[0] - 1*x[1]**2 + 1
         return r

    def jacobian(x):
        j = np.empty((2, 2))
        j[0, 0] = 4*x[0]
        j[0, 1] = 3
        j[1, 0] = 3
        j[1, 1] = -2*x[1]
        return j

    niter = 20  # number of iterations
    x_real = np.array([1, 2])  # actual solution
    x0 = np.array([4, -1])  # initial guess 
    solutions = np.zeros((niter+1, 2), dtype=float)
    solutions[0, :] = x0

    sol = NewtonSolver(residual, jacobian, [0, 3], np.linalg.solve)
    for i in range(niter):
        sol.step(omega=0.8)
        solutions[i+1, :] = sol.x.copy()

    plt.figure('Convergence')
    plt.semilogy(np.arange(niter)+1, sol.fluct)
    plt.xlabel('Iteration number')
    plt.ylabel(r'Fluctuation $||\Delta x||/||x||$')

    plt.figure('x[0]')
    plt.plot(np.arange(niter+1), solutions[:, 0], 'bx--')
    plt.axhline(x_real[0], color='r', ls=':')
    plt.ylabel('Approximation')
    plt.xlabel('Iteration number')

    plt.figure('x[1]')
    plt.plot(np.arange(niter+1), solutions[:, 1], 'bx--')
    plt.axhline(x_real[1], color='r', ls=':')
    plt.ylabel('Approximation')
    plt.xlabel('Iteration number')
