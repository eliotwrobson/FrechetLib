#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import heapq

import numpy as np
import scipy
from numba import njit


def linear_frechet(p: np.ndarray, q: np.ndarray) -> np.float64:
    n_p = p.shape[0]
    n_q = q.shape[0]
    norms = scipy.spatial.distance.cdist(p, q)
    return _linear_frechet(n_p, n_q, norms)


@njit
def _linear_frechet(n_p: int, n_q: int, norms: np.ndarray) -> np.float64:
    """
    From:
    https://github.com/joaofig/discrete-frechet/blob/master/recursive-vs-linear.ipynb
    """
    ca = np.zeros((n_p, n_q), dtype=np.float64)

    prev_p = np.zeros((n_p, n_q), dtype=np.int32)
    prev_q = np.zeros((n_p, n_q), dtype=np.int32)

    for i in range(n_p):
        for j in range(n_q):
            d = norms[i, j]

            if i > 0 and j > 0:
                min_elem = np.inf
                min_x = -1
                min_y = -1

                for prev_i, prev_j in ((i - 1, j), (i, j - 1), (i - 1, j - 1)):
                    prev_val = ca[prev_i, prev_j]
                    if prev_val < min_elem:
                        min_elem = prev_val
                        min_x = prev_i
                        min_y = prev_j

                prev_p[i, j] = min_x
                prev_q[i, j] = min_y

                ca[i, j] = max(min_elem, d)
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i - 1, 0], d)
                prev_p[i, j] = i - 1
                prev_q[i, j] = 0
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[0, j - 1], d)
                prev_p[i, j] = 0
                prev_q[i, j] = j - 1
            else:
                ca[i, j] = d

    return ca[n_p - 1, n_q - 1]


@njit
def linear_frechet_2(p: np.ndarray, q: np.ndarray) -> np.float64:
    """
    Combines above reference implementation with ideas from:
    https://numba.discourse.group/t/dijkstra-on-grid/1483
    """

    n_p = p.shape[0]
    n_q = q.shape[0]

    d = np.linalg.norm(p[0] - q[0])
    priority_queue = [(d, 0, 0)]
    # heapq.heappush(priority_queue, (d, (0, 0)))

    longest_dist = d
    seen = {(0, 0)}
    # ca = np.zeros((n_p, n_q), dtype=np.float64)

    dxys = [(0, 1), (1, 0), (1, 1)]

    while priority_queue:
        curr_dist, curr_x, curr_y = heapq.heappop(priority_queue)

        longest_dist = max(curr_dist, longest_dist)

        if curr_x == n_p - 1 and curr_y == n_q - 1:
            return longest_dist

        for dx, dy in dxys:
            x_new = curr_x + dx
            y_new = curr_y + dy
            pair = (x_new, y_new)

            if not ((0 <= x_new < n_p) and (0 <= y_new < n_q)) or pair in seen:
                continue

            d = np.linalg.norm(p[x_new] - q[y_new])
            seen.add(pair)
            heapq.heappush(priority_queue, (d, x_new, y_new))

    raise ValueError
