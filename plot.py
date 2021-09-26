"""
Plot test-statistic distribution
================================
"""

import numpy as np
import matplotlib.pyplot as plt

from stats import pdf_half_chi2
from data import n_channels


if __name__ == "__main__":

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{fourier}\usepackage{amsmath}')
    plt.rc('font', **{'family':'serif', 'size': 16})

    test_statistic_global = np.load("data/test_statistic_global.npy")
    test_statistic_local = np.load("data/test_statistic_local.npy")

    plt.hist(test_statistic_local, bins='auto', density=True,
             histtype="stepfilled", alpha=0.75, color="SeaGreen", label=r"Fixed mass, $m=151\,\text{GeV}$")
    plt.hist(test_statistic_global, bins='auto', density=True,
             histtype="step", lw=2, label="Floating mass")

    x = np.linspace(0, 25, 10000)
    plt.plot(x, pdf_half_chi2(x, 1), lw=2, label=r"$\frac12 \chi^2$ --- used in ref.~[1]")
    plt.plot(x, pdf_half_chi2(x, n_channels), lw=2, color="DarkGreen", label=r"$\frac12 \chi^2_6$")

    plt.xlabel("$\lambda$")
    plt.ylabel("PDF")
    plt.yticks([])
    plt.ylim(0, 0.5)
    plt.xlim(-1, 20)

    plt.legend()
    plt.savefig("dist.pdf")
