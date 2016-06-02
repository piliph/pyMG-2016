# coding=utf-8
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import functools as ft
import scipy.integrate as spint
from transfer_tools import to_dense


class Bunch:
    """
    Create an object(Bunch) with some Attributes you initialize in the beginning.

    """

    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def prepare_function(f):
    """
    This prepares a function for a simple use of numpy arrays to get a vector f(numpy.array) = numpy.array

    :param f: A function
    :return: Prepared function
    """
    return np.vectorize(f)


def matrix_power(A, n):
    if n == 1:
        return A
    elif n < 1:
        return np.zeros(A.shape)
    else:
        return matrix_power(A, n - 1)
def extract_vector(f, u):
    """
    We expect a function which has arbitrary many arguments f(t,x,y,z, ...),
    the convention is that the first is argument is always time

    :param f: a function f(t,x,y,z,...)
    :param u: u is a list of numpy.arrays for each dimension containing the values one considers to evaluate
    :return: vector with the evaluated rhs
    """
    if len(u) is 1:
        return prepare_function(f)(u[0])
    else:
        vect_list = []
        next_u = u[1:]
        for x in u[0]:
            next_f = ft.partial(f, x)
            vect_list.append(extract_vector(next_f, next_u))
        return np.concatenate(vect_list)


def distributeToFirst(v, N):
    """
    Distribute to first, fill up with zeros
    :param v: numpy vector
    :param N: number of times shape is used
    :return: V=(v,0,...,0)
    """
    z = np.zeros(v.shape)
    vlist = [v] + [z] * (N - 1)
    return np.concatenate(vlist)


def distributeToAll(v, N):
    """
    Distribute to all
    :param v: numpy vector
    :param N: number of times shape is repeated
    :return: V=(v,v,...,v)
    """
    vlist = [v] * (N)
    return np.concatenate(vlist)


def transform_to_unit_interval(x, t_l, t_r):
    return (x - t_l) / (t_r - t_l)


def sparse_inv(P, v):
    """
    Uses sparse solver to compute P^-1 * v
    :param P: sparse_matrix
    :param v: dense vector
    :return:
    """
    return spla.spsolve(P, v)


# for the periodic case
def transformation_matrix_fourier_basis(N):
    psi = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            psi[i, j] = np.exp(2 * np.pi * 1.0j * j * i / N)
    return psi / np.sqrt(N)


# for the dirichlet case
def transformation_matrix_sinus_basis(N):
    psi = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            psi[i, j] = np.sin(np.pi * (j + 1) * (i + 1) / (N + 1))
    return psi


# a permutated fourier basis
def transformation_matrix_harmonic_basis(N):
    psi = np.zeros((N, N), dtype=np.complex128)

    if N % 2 == 1:
        half = (N - 1) / 2 + 1
    else:
        half = N / 2
    for i in range(N):
        for j in range(half):
            psi[j, i] = np.exp(-2 * np.pi * 1.0j * (j + 1) * i / N)
            psi[j + half, i] = np.exp(-2 * np.pi * 1.0j * (N - j - 1) * i / N)
    return psi / np.sqrt(N)


def gamma_k(P, k, m):
    c_s = P[1, 1:m + 1]
    N = P.shape[0]
    my_cos = lambda k, l: np.cos(4 * np.pi / N * k * (l + 0.5))
    return np.dot(c_s, my_cos(k, np.arange(m))) * 2


# analytically computed diagonals
def prolongation_diagonals(P, m):
    N = P.shape[0]
    my_cos = lambda k, l: np.cos(4 * np.pi / N * k * (l + 0.5))
    gamma_k = lambda k: np.dot(P[1, 1:m + 1], my_cos(k, np.arange(m))) * 2
    ak_s = map(lambda i: (1 + gamma_k(i)) * np.sqrt(2) / 2, range(N / 2))
    bk_s = map(lambda i: (1 - gamma_k(i)) * np.sqrt(2) / 2, range(N / 2))
    return ak_s, bk_s


def prolongation_restriction_fourier_diagonals(P, R):
    N = P.shape[0]
    fine_psi = transformation_matrix_fourier_basis(N)
    fine_psi_inv = np.conj(fine_psi.T)
    coarse_psi = transformation_matrix_fourier_basis(N / 2)
    coarse_psi_inv = np.conj(coarse_psi.T)
    transformed_P = np.dot(fine_psi_inv, np.dot(P, coarse_psi))
    transformed_R = np.dot(coarse_psi_inv, np.dot(R, fine_psi))
    a_ks = np.real(np.diag(transformed_P[0:N / 2, :]))
    b_ks = np.real(np.diag(transformed_P[N / 2:, :]))
    d_ks = np.real(np.diag(transformed_R[:, :N / 2]))
    e_ks = np.real(np.diag(transformed_R[:, N / 2:]))
    return a_ks, b_ks, d_ks, e_ks


# DG matrices

def matrixM_tau(basis, domain=[0, 1]):
    order = len(basis)
    M = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            func = lambda t: basis[i](t) * basis[j](t)
            M[i, j] = spint.quad(func, domain[0], domain[1])[0]
    return M


def matrixK_tau(basis, deriv_basis, domain=[0, 1]):
    order = len(basis)
    K = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            func = lambda t: basis[i](t) * deriv_basis[j](t)
            K[i, j] = - spint.quad(func, domain[0], domain[1])[0] + basis[i](domain[1]) * basis[j](domain[1])
    return K


def matrixN_tau(basis, last_basis, domain=[0, 1]):
    order = len(basis)
    N = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            N[i, j] = last_basis[j](domain[0]) * basis[i](domain[0])
    return N


def check_circularity(A):
    # untested
    cmp_line = to_dense(A[0, :])
    for i in range(A.shape[0]):
        sm = np.sum(cmp_line - np.roll(to_dense(A[i, :]), i))
        if np.abs(sm) > 1e-9:
            return False

    return True


def combine_two_P_inv(M, P_1, P_2):
    """ combines two iterative solvers, by returning the combined
        preconditioner of the iterative solver.
    :param M:
    :param P_1:
    :param P_2:
    :return:
    """
    return P_1 + P_2 - P_2.dot(M.dot(P_1))


def combine_N_P_inv(M, P_inv_list):
    if len(P_inv_list) == 1:
        return P_inv_list[0]
    elif len(P_inv_list) == 2:
        return combine_two_P_inv(M, P_inv_list[0], P_inv_list[1])
    else:
        return combine_two_P_inv(M, P_inv_list[0], combine_N_P_inv(M, P_inv_list[1:]))


def get_iteration_matrix_v_cycle(problem_list, pre_smoother_list, post_smoother_list, transfer_list, nu1=1, nu2=1):
    """ Uses the attached smoothers and transferclasses to compute the iteration matrix recursively
    :return: a iteration matrix and the preconditioner
    """
    if len(problem_list) == 1:
        return problem_list[0].A - problem_list[0].A, spla.inv(problem_list[0].A)
    else:
        N = problem_list[0].ndofs
        I = sp.eye(N, format='csc')
        Pinv_c = get_iteration_matrix_v_cycle(problem_list[1:],
                                              pre_smoother_list[1:],
                                              post_smoother_list[1:],
                                              transfer_list[1:])[1]

        CG_P_inv = transfer_list[0].I_2htoh.dot(Pinv_c.dot(transfer_list[0].I_hto2h))
        CG_correction = I - CG_P_inv.dot(problem_list[0].A)

        if nu1 == 0:
            pre_smooth = I
        else:
            pre_smooth = combine_N_P_inv(problem_list[0].A, [pre_smoother_list[0].Pinv] * nu1)

        if nu2 == 0:
            post_smooth = I
        else:
            post_smooth = combine_N_P_inv(problem_list[0].A, [post_smoother_list[0].Pinv] * nu2)

        pre = I - pre_smooth.dot(problem_list[0].A)
        post = I - post_smooth.dot(problem_list[0].A)

        it_matrix = pre.dot(CG_correction.dot(post))
        precond_inv = combine_N_P_inv(problem_list[0].A,
                                      [pre_smooth, CG_P_inv, post_smooth])

        return it_matrix, precond_inv


if __name__ == "__main__":
    # Test the iteration matrix generation with a working multigrid

    from project.poisson1d import Poisson1D
    from project.weighted_jacobi import WeightedJacobi
    # from project.gauss_seidel import GaussSeidel
    from project.linear_transfer import LinearTransfer
    from project.mymultigrid import MyMultigrid
    from pymg.problem_base import ProblemBase
    from project.solversmoother import SolverSmoother

    norm_type = 2
    ntests = 1
    ndofs = 15
    iter_max = 10
    # nlevels = int(np.log2(ndofs + 1))
    nlevels = 2
    prob = Poisson1D(ndofs=ndofs)
    mymg = MyMultigrid(ndofs=ndofs, nlevels=nlevels)
    mymg.attach_transfer(LinearTransfer)
    mymg.attach_smoother(WeightedJacobi, prob.A, omega=2.0 / 3.0)
    k = 6
    xvalues = np.array([(i + 1) * prob.dx for i in range(prob.ndofs)])
    prob.rhs = (np.pi * k) ** 2 * np.sin(np.pi * k * xvalues)
    uex = spla.spsolve(prob.A, prob.rhs)
    res = 1
    err = []
    err_so = []  # Smoother only error
    u = np.zeros(uex.size)
    u_so = np.zeros(uex.size)
    w_jac = mymg.smoo[0]
    # do the multigrid iterationssteps along side with the
    # smoother only steps

    for i in range(iter_max):
        u = mymg.do_v_cycle(u, prob.rhs, 2, 2, 0)
        u_so = w_jac.smooth(prob.rhs, u_so)
        res = np.linalg.norm(prob.A.dot(u) - prob.rhs, np.inf)
        err.append(np.linalg.norm(u - uex, norm_type))
        err_so.append(np.linalg.norm(u_so - uex, norm_type))

    print res, err[-1], err_so[-1]

    # build P_inv for different numbers of smoothing steps

    w_jac_p_inv_list = map(lambda i: combine_N_P_inv(prob.A, [w_jac.Pinv] * i), range(1, 11))

    # check if the P_invs are working correctly by computing the error
    u_so = np.zeros(uex.size)
    err_so_mult_pinv = map(lambda Pinv: np.linalg.norm(uex - u_so - Pinv.dot(prob.rhs - prob.A.dot(u_so)), norm_type),
                           w_jac_p_inv_list)
    print np.asarray(err_so) - np.asarray(err_so_mult_pinv)
    # works fine

    # build the iteration matrix for the v_cycle
    transfer_list = mymg.trans
    pre_smooth_list = mymg.smoo
    post_smooth_list = mymg.smoo
    problem_list = []
    new_ndofs = ndofs
    A_coarse = prob.A
    rhs = prob.rhs
    for l in range(nlevels - 1):
        # this can't work because Galerkin is used
        # problem_list.append(Poisson1D(ndofs=new_ndofs))
        problem_list.append(ProblemBase(rhs.size, A_coarse, rhs))
        print transfer_list[l].I_hto2h.shape
        rhs = transfer_list[l].I_hto2h.dot(rhs)
        A_coarse = transfer_list[l].I_hto2h.dot(A_coarse.dot(transfer_list[l].I_2htoh))
    problem_list.append(ProblemBase(rhs.size, A_coarse, rhs))
    post_smooth_list.append(SolverSmoother(problem_list[-1].A))
    pre_smooth_list.append(SolverSmoother(problem_list[-1].A))

    it_matrix, precond_inv = get_iteration_matrix_v_cycle(problem_list,
                                                          pre_smooth_list,
                                                          post_smooth_list,
                                                          transfer_list, 2, 2)
    # use iteration matrix to compute the error
    err_tmp = np.zeros(uex.size) - uex
    err_mat = np.zeros(len(err) + 1)
    for i in range(len(err) + 1):
        err_mat[i] = np.linalg.norm(err_tmp, norm_type)
        err_tmp = it_matrix.dot(err_tmp)
    # print err, err_mat
    print err - err_mat[1:]
    # aktueller stand bei 2 leveln funktionierts
    # bei drei leveln nicht mehr
