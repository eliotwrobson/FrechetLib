#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import heapq

import numpy as np
import scipy
from numba import njit  # type: ignore[attr-defined]

_DiscreteReturnT = tuple[np.float64, list[tuple[int, int]]]


def discrete_frechet(p: np.ndarray, q: np.ndarray) -> _DiscreteReturnT:
    n_p = p.shape[0]
    n_q = q.shape[0]
    norms = scipy.spatial.distance.cdist(p, q)
    return _discrete_frechet(n_p, n_q, norms)


@njit
def _discrete_frechet(n_p: int, n_q: int, norms: np.ndarray) -> _DiscreteReturnT:
    """
    From:
    https://github.com/joaofig/discrete-frechet/blob/master/recursive-vs-linear.ipynb
    """
    ca = np.zeros((n_p, n_q), dtype=np.float64)

    prev = np.zeros((n_p, n_q, 2), dtype=np.int32)

    for i in range(n_p):
        for j in range(n_q):
            d = norms[i, j]

            if i > 0 and j > 0:
                min_elem = np.inf
                min_idx = (-1, -1)

                for prev_idx in ((i - 1, j), (i, j - 1), (i - 1, j - 1)):
                    prev_val = ca[prev_idx]
                    if prev_val < min_elem:
                        min_elem = prev_val
                        min_idx = prev_idx

                prev[i, j] = min_idx

                ca[i, j] = max(min_elem, d)
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i - 1, 0], d)
                prev[i, j] = i - 1, 0
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[0, j - 1], d)
                prev[i, j] = 0, j - 1
            elif i == 0 and j == 0:
                ca[i, j] = d
            else:
                raise Exception

    # Reconstructing the solution
    curr_x = n_p - 1
    curr_y = n_q - 1
    morphing = [(curr_x, curr_y)]
    while curr_x != 0 or curr_y != 0:
        curr_x, curr_y = prev[curr_x, curr_y]
        morphing.append((curr_x, curr_y))

    morphing.reverse()

    return ca[n_p - 1, n_q - 1], morphing


@njit
def discrete_retractable_frechet(p: np.ndarray, q: np.ndarray) -> _DiscreteReturnT:
    """
    Combines above reference implementation with ideas from:
    https://numba.discourse.group/t/dijkstra-on-grid/1483
    """

    n_p = p.shape[0]
    n_q = q.shape[0]

    if n_p == 0 or n_q == 0:
        raise ValueError

    d = np.linalg.norm(p[0] - q[0])
    priority_queue = [(d, 0, 0)]

    longest_dist = d
    prev = {(0, 0): (-1, -1)}

    dxys = [(0, 1), (1, 0), (1, 1)]

    while priority_queue:
        curr_dist, curr_x, curr_y = heapq.heappop(priority_queue)

        longest_dist = max(curr_dist, longest_dist)

        if curr_x == n_p - 1 and curr_y == n_q - 1:
            break

        for dx, dy in dxys:
            x_new = curr_x + dx
            y_new = curr_y + dy
            pair = (x_new, y_new)

            if not ((0 <= x_new < n_p) and (0 <= y_new < n_q)) or pair in prev:
                continue

            d = np.linalg.norm(p[x_new] - q[y_new])
            prev[pair] = (curr_x, curr_y)
            heapq.heappush(priority_queue, (d, x_new, y_new))

    # Reconstructing the solution
    curr = (n_p - 1, n_q - 1)
    morphing = [curr]
    while curr != (0, 0):
        curr = prev[curr]
        morphing.append(curr)

    morphing.reverse()

    return longest_dist, morphing
