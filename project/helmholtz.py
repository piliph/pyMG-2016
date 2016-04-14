# coding=utf-8
import numpy as np
import scipy.sparse as sp

from pymg.problem_base import ProblemBase


class Helmholtz(ProblemBase):
    """Implementation of the 1D Helmholtz problem.

    Here we define the 1D Helmholtz problem :math:`-\Delta u - \sigma u = 0` with
    Dirichlet-Zero boundary conditions. This is the homogeneous problem,
    derive from this class if you want to play around with different RHS.

    Attributes:
        dx (float): mesh size
    """
    def __init__(self, sigma, ndofs, *args, **kwargs):
        """Initialization routine for the Helmholtz1D problem

        Args:
            ndofs (int): number of degrees of freedom (see
                :attr:`pymg.problem_base.ProblemBase.ndofs`)
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        self.dx = 1.0 / (ndofs + 1)
        # compute system matrix A, scale by 1/dx^2
        A = self.__get_system_matrix(sigma, ndofs, self.dx)
        rhs = self.__get_rhs(ndofs)

        super(Helmholtz, self).__init__(ndofs, A, rhs, *args, **kwargs)

    @staticmethod
    def __get_system_matrix(sigma, ndofs, dx):
        """Helper routine to get the system matrix discretizing :math:`-Delta` with second order FD

        Args:
            sigma (float): damping Argument
            ndofs (int): number of inner grid points (no boundaries!)
        Returns:
            scipy.sparse.csc_matrix: sparse system matrix A
                of size :attr:`ndofs` x :attr:`ndofs`
        """
        data = np.array([[2./dx**2-sigma]*ndofs, [-1./dx**2-sigma]*ndofs, [-1./dx**2-sigma]*ndofs])
        diags = np.array([0, -1, 1])
        return sp.spdiags(data, diags, ndofs, ndofs, format='csc')

    @staticmethod
    def __get_rhs(ndofs):
        """Helper routine to set the right-hand side

        Args:
            ndofs (int): number of inner grid points (no boundaries!)
        Returns:
            numpy.ndarray: the right-hand side vector of size :attr:`ndofs`
        """
        return np.zeros(ndofs)

    @property
    def u_exact(self):
        """Routine to compute the exact solution

        Returns:
            numpy.ndarray: exact solution array of size :attr:`ndofs`
        """
        return np.zeros(self.ndofs)
