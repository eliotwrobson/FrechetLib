from typing import Any

import numba as nb
import numba.typed as nbt
import numpy as np

import frechetlib.frechet_utils as fu
import frechetlib.retractable_frechet as rf


@nb.njit
def frechet_width_approx(
    P: np.ndarray, idx_range: tuple[int, int] | None = None
) -> float:
    """
    2-approximation to the Frechet distance between
    P[first(rng)]-P[last(rng)] and he polygon
    P[rng]
    Here, rng is a range i:j
    """
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


@nb.njit
def frechet_mono_via_refinement(
    P: np.ndarray, Q: np.ndarray, approx: float
) -> tuple[fu.Morphing, bool]:
    """
    Computes the "true" monotone Frechet distance between P and Q,
    using the ve_r algorithm. It does refinement, to add vertices if
    needed to get monotonicity. Note, that there is an eps parameter -
    for real inputs you can set it quite small. In any case, in the
    worst case it only approximates the Frechet (monotone)
    distance. It returns a morphing, and a boolean that is true if the
    result is the true Frechet distance.

    Observe, that this function might be too slow if the two curves
    are huge, since it does not simplify them before computing the
    distance.
    """

    assert 1.0 <= approx

    ve_morphing = rf.retractable_ve_frechet(P, Q)

    monotone_morphing = ve_morphing.copy()
    monotone_morphing.make_monotone()

    # Continue until monotone_morphing.dist <= approx * ve_morphing.dist
    while monotone_morphing.dist > approx * ve_morphing.dist:
        # Add points where monotonicity was broken to improve distance
        new_P, new_Q = add_points_to_make_monotone(ve_morphing)

        # Compute new ve frechet distance for curves
        ve_morphing = rf.retractable_ve_frechet(new_P, new_Q)

        # Make monotone
        monotone_morphing = ve_morphing.copy()
        monotone_morphing.make_monotone()

    return monotone_morphing, np.isclose(ve_morphing.dist, monotone_morphing.dist)


# TODO have to write test cases for this
@nb.njit
def add_points_to_make_monotone(
    morphing: fu.Morphing,
) -> tuple[np.ndarray, np.ndarray]:
    # TODO add intermediate vertices here
    # Doing the same here:
    # https://github.com/sarielhp/FrechetDist.jl/blob/main/src/frechet.jl#L626

    P = morphing.P
    Q = morphing.Q
    morphing_list = morphing.morphing_list

    # First, add points to P
    new_P = []
    k = 0
    while k < len(morphing_list):
        # Vertex-vertex event, can skip
        if morphing_list[k].i_is_vert:
            new_P.append(P[morphing_list[k].i])
            old_k = k
            while (
                k < len(morphing_list)
                and morphing_list[old_k].i == morphing_list[k].i
                and morphing_list[k].i_is_vert
            ):
                k += 1
            continue

        loc = morphing_list[k].i
        events = []

        # [old_k,k) is the indices of points that are on the same segment
        # So increase new_k to get the max window where this is the case
        while (
            k < len(morphing_list)
            and not morphing_list[k].i_is_vert
            and morphing_list[k].i == loc
        ):
            events.append(morphing_list[k])
            k += 1

        # Next, check if the offsets are monotone as-given
        monotone = True
        for j in range(len(events) - 1):
            monotone = monotone and (events[j].t <= events[j + 1].t)

        # TODO double check this is the right thing to do
        if monotone:
            continue

        events = sorted(events, key=fu.eid_get_coefficient)

        for j in range(len(events)):
            new_P.append(events[j].p)

            if not monotone and j < len(events) - 1:
                new_P.append((events[j].p + events[j + 1].p) / 2)

    # # Next, add points to Q, same as above but hard to share logic
    new_Q = []
    k = 0
    while k < len(morphing_list):
        if morphing_list[k].j_is_vert:
            new_Q.append(Q[morphing_list[k].j])
            old_k = k
            while (
                k < len(morphing_list)
                and morphing_list[old_k].j == morphing_list[k].j
                and morphing_list[k].j_is_vert
            ):
                k += 1
            continue

        loc = morphing_list[k].j
        events = []

        # [old_k,k) is the indices of points that are on the same segment
        # So increase new_k to get the max window where this is the case
        while (
            k < len(morphing_list)
            and not morphing_list[k].j_is_vert
            and morphing_list[k].j == loc
        ):
            events.append(morphing_list[k])
            k += 1

        # Next, check if the offsets are monotone as-given
        monotone = True
        for j in range(len(events) - 1):
            monotone = monotone and (events[j].t <= events[j + 1].t)

        if monotone:
            continue

        events = sorted(events, key=fu.eid_get_coefficient)
        print("old: ", new_Q)
        for j in range(len(events)):
            new_Q.append(events[j].p)

            if not monotone and j < len(events) - 1:
                new_Q.append((events[j].p + events[j + 1].p) / 2)

    # Finally, assemble into output arrays
    new_P_final = np.empty((len(new_P), new_P[0].shape[0]))
    new_Q_final = np.empty((len(new_Q), new_Q[0].shape[0]))

    for k in range(len(new_P)):
        new_P_final[k] = new_P[k]

    for k in range(len(new_Q)):
        new_Q_final[k] = new_Q[k]

    return new_P_final, new_Q_final


@nb.njit
def simplify_polygon_radius(P: np.ndarray, r: float) -> tuple[np.ndarray, list[int]]:
    curr = P[0]
    indices = [0]
    n = P.shape[0]

    for i in range(1, n):
        if np.linalg.norm(P[i] - curr) > r:
            curr = P[i]
            indices.append(i)

    indices.append(n - 1)

    new_P = np.empty((len(indices), P.shape[0]))

    for k in range(len(indices)):
        new_P[k] = P[indices[k]]

    return new_P, indices


@nb.njit
def frechet_c_mono_approx_subcurve(
    P: np.ndarray, P_subcurve: np.ndarray, p_indices: list[int]
) -> tuple[float, list[fu.EID]]:
    """
    Approximates the Frechet distance between a curve (P) and subcurve
    (P_subcurve). Here, P_subcurve vertices are the vertices of P
    specified by p_indices. That is P_subcurve[i] = P[p_indices[i]].
    """

    res = []
    width = 0
    for i in range(len(p_indices) - 1):
        curr_idx = p_indices[i]
        next_idx = p_indices[i + 1]

        next_event = fu.from_curve_indices(i, True, curr_idx, True, P, P_subcurve)
        width = max(width, next_event.dist)
        res.append(next_event)

        for j in range(curr_idx + 1, next_idx):
            next_event = fu.from_curve_indices(i, True, j, False, P, P_subcurve)
            width = max(width, next_event.dist)
            res.append(next_event)

    next_event = fu.from_curve_indices(
        len(P) - 1, True, len(P_subcurve) - 1, True, P, P_subcurve
    )
    width = width = max(width, next_event.dist)
    res.append(width)

    return res


def frechet_c_approx(P: np.ndarray, Q: np.ndarray, approx_ratio: float) -> Any:
    """
    Approximates the continuous Frechet distance between the two input
    curves. Returns a monotone morphing realizing it.

    # Arguments

    - `approx` : The output morhing has Frechet distance <= approx*optimal.

    Importantly, approx can be larger than 2, if you want a really
    rough approximation.
    """
    P_orig = P
    Q_orig = Q

    # Modeled after:
    # https://github.com/sarielhp/FrechetDist.jl/blob/main/src/frechet.jl#L810
    upper_bound_dist = frechet_dist_upper_bound(P, Q)

    # radius of simplification allowed
    r = upper_bound_dist / (approx_ratio + 4.0)

    while r >= (upper_bound_dist / (approx_ratio + 4.0)):
        r /= 2.0
        P = simplify_polygon_radius(P, r)
        Q = simplify_polygon_radius(Q, r)

        morphing, _ = frechet_mono_via_refinement(P, Q, (3.0 + approx_ratio) / 4.0)

    p_width, morphing_p = frechet_c_mono_approx_subcurve(P_orig, P, p_indices)
    q_width, morphing_q = frechet_c_mono_approx_subcurve(Q_orig, Q, q_indices)

    _, first_combined = fu.morphing_combine(P_orig, P, Q, morphing_p, morphing)
    _, first_combined_monotone = get_monotone_morphing_width(
        nbt.List(first_combined), P, Q
    )
    width, final_combined = fu.morphing_combine(
        P_orig, Q, Q_orig, first_combined_monotone, morphing_q
    )

    ratio = width / (frechet_distance - 2.0 * max(p_width, q_width))

    # TODO might be need to run through this a second time in a loop? Not sure
    return width, ratio, final_combined


def frechet_c_compute(P: np.ndarray, Q: np.ndarray, f_accept_appx: bool = True):

    width, ratio, morphing = frechet_c_approx(P, Q, 2.0)
    approx_refinement = 1.001
    approx_ratio = min(1.0 + (P.shape[0] + Q.shape[0]) / (100.0 * width), 1.1)

    # TODO fill this if case with the appx ratio from frechet_c_approx
    if False:
        new_morphing = morphing
        ratio = 2.0
    else:
        width, ratio, new_morphing = frechet_c_approx(P, Q, approx_ratio)
        # ratio

    Pl, Ql = fu.extract_vertex_radii(P, Q, new_morphing)
    lower_bound = width / ratio

    factor = 4.0
    while True:
        Pz = (lower_bound - Pl) / factor
        Qz = (lower_bound - Ql) / factor

        p_indices = fu.simplify_polygon_radii(P, Pz)
        q_indices = fu.simplify_polygon_radii(Q, Qz)

        Ps = np.take(P, p_indices, axis=0)
        Qs = np.take(Q, q_indices, axis=0)

        PSR, QSR, morphing, dist, is_exact = frechet_mono_via_refinement(
            Ps, Qs, approx_refinement
        )
        # TODO continue writing with
        # https://github.com/sarielhp/FrechetDist.jl/blob/main/src/frechet.jl#L1057
        # m_a = frechet_ve_r_mono_compute(poly_a, PSR)
        # mmu = Morphing_combine(m_a, m_mid)
        # m_b = frechet_ve_r_mono_compute(QSR, poly_b)
        # mw = Morphing_combine(mmu, m_b)
