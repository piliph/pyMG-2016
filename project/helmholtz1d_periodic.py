# coding=utf-8
import numpy as np
import scipy.sparse as sp

from pymg.problem_base import ProblemBase


class Helmholtz1D_Periodic(ProblemBase):
    """Implementation of the 1D Helmholtz problem.

    Here we define the 1D Poisson problem :math:`-\Delta u - \sigma u = 0` with
    Dirichlet-Zero boundary conditions. This is the homogeneous problem,
    derive from this class if you want to play around with different RHS.

    Attributes:
        dx (float): mesh size
    """

    def __init__(self, ndofs, sigma=1, *args, **kwargs):
        """Initialization routine for the Poisson1D problem

        Args:
            ndofs (int): number of degrees of freedom (see
                :attr:`pymg.problem_base.ProblemBase.ndofs`)
            omega (float, optional): wave number
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        self.dx = 1.0 / ndofs
        # compute system matrix A, scale by 1/dx^2
        A = self.__get_system_matrix(ndofs, self.dx, sigma)
        A[0, -1] = A[0, 1]
        A[-1, 0] = A[1, 0]
        A = 1.0 / (self.dx ** 2) * A
        rhs = self.__get_rhs(ndofs)

        super(Helmholtz1D_Periodic, self).__init__(ndofs, A, rhs, *args, **kwargs)

    @staticmethod
    def __get_system_matrix(ndofs, dx, sigma):
        """Helper routine to get the system matrix discretizing the Helmholtz operator
         with second order FD

        Args:
            ndofs (int): number of inner grid points (no boundaries!)
            dx (float): mesh size
            sigma (float): wave number
        Returns:
            scipy.sparse.csc_matrix: sparse system matrix A
                of size :attr:`ndofs` x :attr:`ndofs`
        """
        data = np.array([[2 - sigma * dx ** 2] * ndofs, [-1] * ndofs, [-1] * ndofs])
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

        # @property
        # def u_exact(self):
        #     """Routine to compute the exact solution
        #
        #     Returns:
        #         numpy.ndarray: exact solution array of size :attr:`ndofs`
        #     """
        #     return np.zeros(self.ndofs)
