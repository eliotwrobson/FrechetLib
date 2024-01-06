#! /usr/bin/env python2
# -*- coding: utf-8 -*-

from typing import Callable

import numpy as np
from numba import jit


def _c(ca, i, j, p, q):
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = np.linalg.norm(p[i] - q[j])
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i - 1, 0, p, q), np.linalg.norm(p[i] - q[j]))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j - 1, p, q), np.linalg.norm(p[i] - q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(
            min(
                _c(ca, i - 1, j, p, q),
                _c(ca, i - 1, j - 1, p, q),
                _c(ca, i, j - 1, p, q),
            ),
            np.linalg.norm(p[i] - q[j]),
        )
    else:
        ca[i, j] = float("inf")

    return ca[i, j]


def discrete_frechet_distance(P: np.ndarray, Q: np.ndarray) -> np.float64:
    p = np.array(P, np.float64)
    q = np.array(Q, np.float64)

    len_p = len(p)
    len_q = len(q)

    if len_p == 0 or len_q == 0:
        raise ValueError("Input curves are empty.")

    # if len_p != len_q or len(p[0]) != len(q[0]):
    #    raise ValueError("Input curves do not have the same dimensions.")

    ca = np.ones((len_p, len_q), dtype=np.float64) * -1

    dist = _c(ca, len_p - 1, len_q - 1, p, q)
    return dist


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
