"""
Functions for implementing Gross-Vitells method and half chi-squared distribution
=================================================================================
"""

import numpy as np
from scipy.stats import chi2, norm
from scipy.special import comb, gamma


def sf_half_chi2(level, dof):
    """
    @returns Survival function for sum of half-chi-squared distributions
    """
    terms = np.zeros_like(level)
    for i in np.arange(1, dof + 1):
        terms += 0.5**dof * comb(dof, i) * chi2.sf(level, i)
    return terms

def pdf_half_chi2(level, dof):
    """
    @returns PDF for sum of half-chi-squared distributions
    """
    terms = np.zeros_like(level)
    for i in np.arange(1, dof + 1):
        terms += 0.5**dof * comb(dof, i) * chi2.pdf(level, i)
    return terms

def p_local(level, dof):
    return sf_half_chi2(level, dof)

def z_from_p(p):
    return norm.isf(p)

def euler_function(dof, level):
    """
    @returns Characteristic Euler function for chi-squared with particular dof

    See \rho_1(c) on p29
    https://arxiv.org/pdf/1803.03858.pdf
    """
    return np.sqrt(2. / np.pi) * level**(0.5 * (dof - 1)) / gamma(0.5 * dof) * np.exp(-0.5 * level)

def p_global(level, dof, N):
    """
    EC for chi^2_{5/2} is just a mixture of ECs
    https://arxiv.org/pdf/1207.3840.pdf
    eq. 15 and theorem 1.
    """
    i = np.arange(1, dof + 1)
    return p_local(level, dof) + N * 0.5**dof * (comb(dof, i) * euler_function(i, level)).sum()

def euler(line_):
    """
    @returns Euler characteristic of a line
    """
    masked = np.ma.masked_array(line_, line_ == 0)
    return len(np.ma.clump_masked(masked))

def euler_from_lambda(test_statistic, level):
    """
    @returns Euler characteristic for test-statistic
    """
    excursion = np.array(test_statistic <= level).astype(int)
    return euler(excursion)

def coeff(mean_euler, level, dof):
    """
    @returns Coefficient required for calculating global p-value
    """
    local = p_local(level, dof)
    global_ = p_global(level, dof, 1)
    return (mean_euler - local) / (global_ - local)

def p_mc(test_statistic, level):
    """
    @returns P-value from MC simulations
    """
    return (test_statistic >= level).sum() / len(test_statistic)
