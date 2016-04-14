import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import scipy.sparse.linalg as spLA
from project.poisson1d import Poisson1D
from project.gauss_seidel import GaussSeidel

if __name__ == "__main__":

    rc('font', family='sans-serif', size=32)
    rc('legend', fontsize='small')
    rc('xtick', labelsize='small')
    rc('ytick', labelsize='small')

    fig = plt.subplots(figsize=(15, 8))

    ndofs = 63
    prob = Poisson1D(ndofs)
    smoo = GaussSeidel(prob.A)

    xvalues = np.array([(i + 1) * prob.dx for i in range(prob.ndofs)])
    rhs = np.sin(np.pi * 2 * xvalues)
    uex = spLA.spsolve(smoo.A, rhs)

    u = uex
    for i in range(10):
        u = smoo.smooth(rhs, u)
        if i == 1 or i == 10:
            plt.plot(xvalues, u)

    plt.show()

    rhs = np.sin(np.pi * 16 * xvalues)

    u = uex
    for i in range(10):
        u = smoo.smooth(rhs, u)
        if i == 1 or i == 10:
            plt.plot(xvalues, u)

    plt.show()

    rhs = (np.sin(np.pi * 2 * xvalues) + np.sin(np.pi * 16 * xvalues))/2

    u = uex
    for i in range(10):
        u = smoo.smooth(rhs, u)
        if i == 1 or i == 10:
            plt.plot(xvalues, u)

    plt.show()

