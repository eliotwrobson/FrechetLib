#! /usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np


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

    if len_p != len_q or len(p[0]) != len(q[0]):
        raise ValueError("Input curves do not have the same dimensions.")

    ca = np.ones((len_p, len_q), dtype=np.float64) * -1

    dist = _c(ca, len_p - 1, len_q - 1, p, q)
    return dist
