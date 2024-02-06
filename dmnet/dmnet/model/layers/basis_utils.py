import numpy as np
import sympy as sym
from functools import partial

from scipy.special import binom

def b(v, n, x):
    res = binom(n, v) * x ** v * (1 - x) ** (n - v)
    return res

def fcut(r, rcut):
    if rcut >= r:
        return 0
    return np.exp(-r ** 2 / (rcut ** 2 - r ** 2))

def bernstein_polys(K, gamma):
    res = [lambda r: binom(k, K - 1, np.exp(-gamma * r) * fcut(r)) for k in range(K)]
    res = None
    return res