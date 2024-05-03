import heapq as hq

import numpy as np

import frechetlib.frechet_utils as fu
import frechetlib.retractable_frechet as rf


# NOTE Needs to be a separate algorithm from the other
# continuous Frechet ones because the additive property
# could cause non-convergence.
# @njit
def sweep_frechet_compute_refine_mono(P: np.ndarray, Q: np.ndarray) -> fu.Morphing:
    ell = P.shape[0] + Q.shape[0]

    rate_limit = 0.01

    while True:
        morphing = rf.retractable_ve_frechet(P, Q, None, None, True)

        if morphing.is_monotone():
            break

        error = morphing.copy().make_monotone()
        rate = error / ell

        if rate < rate_limit:
            break

        P, Q = fu.add_points_to_make_monotone(morphing)

    return morphing


# @njit
def get_edge_value(i: int, j: int, P: np.ndarray, Q: np.ndarray) -> float:
    n_p = P.shape[0]
    assert 0 <= i < n_p - 1
    weight = np.inf

    if j > 0:
        dist = fu.line_line_distance(P[i], P[i + 1], Q[j - 1], Q[j])
        weight = min(weight, dist)
    if j < Q.shape[0] - 1:
        dist = fu.line_line_distance(P[i], P[i + 1], Q[j], Q[j + 1])
        weight = min(weight, dist)

    length = float(np.linalg.norm(P[i] - P[i + 1]))
    return weight * length


# @njit
def sweep_frechet_compute_lower_bound(P: np.ndarray, Q: np.ndarray) -> float:
    work_queue = [(0.0, 0, 0)]
    prev_weight = {(0, 0): 0.0}

    n_p = P.shape[0]
    n_q = Q.shape[0]

    res = 0.0

    while work_queue:
        weight, i, j = hq.heappop(work_queue)

        res = max(res, weight)

        if i == n_p - 1 and j == n_q - 1:
            break

        if i < n_p - 1:
            new_weight = weight + get_edge_value(i, j, P, Q)
            new_idx = (i + 1, j)
            if prev_weight.get(new_idx, np.inf) > new_weight:
                prev_weight[new_idx] = new_weight
                hq.heappush(work_queue, (new_weight, i + 1, j))

        if j < n_q - 1:
            new_weight = weight + get_edge_value(j, i, Q, P)
            new_idx = (i, j + 1)
            if prev_weight.get(new_idx, np.inf) > new_weight:
                prev_weight[new_idx] = new_weight
                hq.heappush(work_queue, (new_weight, i, j + 1))

    return res
