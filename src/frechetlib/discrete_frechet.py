#! /usr/bin/env python2
# -*- coding: utf-8 -*-

import heapq

import numpy as np
from numba import jit
from numba.typed import Dict, List


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
def linear_frechet_2(p: np.ndarray, q: np.ndarray) -> np.float64:
    """
    Combines above reference implementation with ideas from:
    https://numba.discourse.group/t/dijkstra-on-grid/1483
    """

    n_p = p.shape[0]
    n_q = q.shape[0]

    d = np.linalg.norm(p[0] - q[0])
    priority_queue = List([(d, (0, 0))])
    # heapq.heappush(priority_queue, (d, (0, 0)))

    longest_dist = d
    seen = {(0, 0)}
    # ca = np.zeros((n_p, n_q), dtype=np.float64)

    dxys = [(0, 1), (1, 0), (1, 1)]

    while priority_queue:
        curr_dist, (curr_x, curr_y) = heapq.heappop(priority_queue)

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
            heapq.heappush(priority_queue, (d, pair))

    raise ValueError
