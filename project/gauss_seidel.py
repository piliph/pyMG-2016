# coding=utf-8
import scipy.sparse as sp
import scipy.sparse.linalg as spLA
import numpy as np

from pymg.smoother_base import SmootherBase


class GaussSeidel(SmootherBase):
    """Implementation of the Gauss-Seidel iteration

    Attributes:
        D (scipy.sparse.csc_matrix): Preconditioner
        U (scipy.sparse.csc_matrix): Upper triangular Matrix of A
        L (scipy.sparse.csc_matrix): Lower triangular Matrix of A
        DLinv (scipy.sparse.csc_matrix): inverse of the Preconditioner
    """

    def __init__(self, A, *args, **kwargs):
        """Initialization routine for the smoother

        Args:
            A (scipy.sparse.csc_matrix): sparse matrix A of the system to solve
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super(GaussSeidel, self).__init__(A, *args, **kwargs)

        self.D = sp.spdiags(self.A.diagonal(), 0, self.A.shape[0], self.A.shape[1],
                            format='csc')
        self.L = sp.tril(self.A, k=-1)
        self.U = sp.triu(self.A, k=1)
        # precompute inverse of the preconditioner for later usage
        self.DLinv = spLA.inv(self.D + self.L)

    def smooth(self, rhs, u_old):
        """
        Routine to perform a smoothing step

        Args:
            rhs (numpy.ndarray): the right-hand side vector, size
                :attr:`pymg.problem_base.ProblemBase.ndofs`
            u_old (numpy.ndarray): the initial value for this step, size
                :attr:`pymg.problem_base.ProblemBase.ndofs`

        Returns:
            numpy.ndarray: the smoothed solution u_new of size
                :attr:`pymg.problem_base.ProblemBase.ndofs`
        """
        u_new = self.DLinv.dot(rhs - self.U.dot(u_old))
        return u_new
