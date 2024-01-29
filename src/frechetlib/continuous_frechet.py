import numba as nb
import numpy as np

import frechetlib.frechet_utils as fu


@nb.njit
def frechet_width_approx(
    P: np.ndarray, idx_range: tuple[int, int] | None = None
) -> float:
    n = P.shape[0]

    if idx_range is None:
        start, end = 0, n
    else:
        start, end = idx_range

    start_point = P[start]
    end_point = P[end - 1]

    leash = 0.0
    t = 0.0

    # TODO double check w/ Sariel because this seems like a weird min condition
    for i in range(start + 1, end - 1):
        dist, new_t = fu.line_point_distance(start_point, end_point, P[i])

        if new_t > t:
            t = new_t
            leash = max(leash, dist)

    return leash


@nb.njit
def frechet_dist_upper_bound(
    P: np.ndarray,
    Q: np.ndarray,
) -> float:
    """
    Returns a rough upper bound on the Frechet distance between the two
    curves. This upper bound is on the continuous distance. No guarentee
    on how bad the approximation is. This is used as a starting point for
    real approximation of the Frechet distance, and should not be used
    otherwise.
    """

    w_a = frechet_width_approx(P)
    w_b = frechet_width_approx(Q)

    if P.shape[0] < 2 or Q.shape[0] < 2:
        return w_a + w_b

    w = max(np.linalg.norm(P[0] - Q[0]), np.linalg.norm(P[-1] - Q[-1]))

    return w_a + w_b + w
