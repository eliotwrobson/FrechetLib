#! /usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from numba import jit


@jit(nopython=True)
def linear_frechet(p: np.ndarray, q: np.ndarray) -> float:
    """
    From:
    https://github.com/joaofig/discrete-frechet/blob/master/recursive-vs-linear.ipynb
    """
    n_p = p.shape[0]
    n_q = q.shape[0]
    ca = np.zeros((n_p, n_q), dtype=np.float64)

    for i in range(n_p):
        for j in range(n_q):
            d = np.linalg.norm(p[i] - q[j])

            if i > 0 and j > 0:
                ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d)
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i - 1, 0], d)
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[0, j - 1], d)
            else:
                ca[i, j] = d
    return ca[n_p - 1, n_q - 1]


@jit(nopython=True)
def linear_frechet_2(p: np.ndarray, q: np.ndarray) -> float:
    n_p = p.shape[0]
    n_q = q.shape[0]
    ca = np.zeros((n_p, n_q), dtype=np.float64)

    for i in range(n_p):
        for j in range(n_q):
            d = np.linalg.norm(p[i] - q[j])

            if i > 0 and j > 0:
                ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d)
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i - 1, 0], d)
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[0, j - 1], d)
            else:
                ca[i, j] = d
    return ca[n_p - 1, n_q - 1]
