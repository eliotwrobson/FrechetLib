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

    limit = 3
    k = 0
    # Continue until monotone_morphing.dist <= approx * ve_morphing.dist
    while monotone_morphing.dist > approx * ve_morphing.dist:
        # print(monotone_morphing.dist > approx * ve_morphing.dist,monotone_morphing.dist,approx,ve_morphing.dist,)

        k += 1
        if k >= limit:
            raise Exception
        # Add points where monotonicity was broken to improve distance
        new_P, new_Q = add_points_to_make_monotone(monotone_morphing)

        # Compute new ve frechet distance for curves
        ve_morphing = rf.retractable_ve_frechet(new_P, new_Q)

        # Make monotone
        monotone_morphing = ve_morphing.copy()
        monotone_morphing.make_monotone()

    return monotone_morphing, np.isclose(ve_morphing.dist, monotone_morphing.dist)


# TODO have to write test cases for this, it's buggy and inserting points with nan coordinates
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
                morphing_list[old_k].i == morphing_list[k].i
                and morphing_list[k].i_is_vert
            ):
                k += 1
            continue

        loc = morphing_list[k].i
        offsets = []

        # [old_k,k) is the indices of points that are on the same segment
        # So increase new_k to get the max window where this is the case
        while (
            k < len(morphing_list)
            and not morphing_list[k].i_is_vert
            and morphing_list[k].i == loc
        ):
            offsets.append(morphing_list[k].t)
            k += 1

        # TODO there's an extra 1.0 getting added here for some reason
        # Next, check if the offsets are monotone as-given
        monotone = True
        for j in range(len(offsets) - 1):
            monotone = monotone and (offsets[j] <= offsets[j + 1])

        offsets = sorted(offsets)
        print("old:", new_P)

        for j in range(len(offsets)):
            # TODO if speed is an issue, could be possible to use stored points of P
            new_P.append(fu.convex_comb(P[loc], P[loc + 1], offsets[j]))
            if not monotone and j < len(offsets) - 1:
                new_P.append(
                    fu.convex_comb(
                        P[loc], P[loc + 1], (offsets[j] + offsets[j + 1]) / 2
                    )
                )

        print(k, offsets, monotone, new_P)
        # TODO I think there's an off by one here
        # k = new_k + 1

    # Next, add points to Q, same as above but hard to share logic
    new_Q = []
    k = 0
    while k < len(morphing_list):
        # Vertex-vertex event, can ignore
        if morphing_list[k].j_is_vert:
            new_Q.append(Q[morphing_list[k].j])
            k += 1
            continue

        new_k = k
        loc = morphing_list[k].i
        offsets = []

        while (
            new_k < len(morphing_list) - 1
            and not morphing_list[new_k + 1].j_is_vert
            and morphing_list[new_k + 1].j == loc
        ):
            offsets.append(morphing_list[new_k].t)
            new_k += 1

        # [k:new_k) is the indices of points that are on the same segment
        monotone = True
        offsets = sorted(offsets)
        for j in range(len(offsets) - 1):
            monotone = monotone and (offsets[j] <= offsets[j + 1])

        for j in range(len(offsets)):
            new_P.append(fu.convex_comb(Q[loc], Q[loc + 1], offsets[j]))
            if not monotone and j < len(offsets) - 1:
                new_P.append(
                    fu.convex_comb(
                        Q[loc], Q[loc + 1], (offsets[j] + offsets[j + 1]) / 2
                    )
                )
        # TODO I think there's an off by one here
        k = new_k + 1

    new_P_final = np.empty((len(new_P), new_P[0].shape[0]))
    new_Q_final = np.empty((len(new_Q), new_Q[0].shape[0]))

    # Finally, make the arrays
    for k in range(len(new_P)):
        new_P_final[k] = new_P[k]

    for k in range(len(new_Q)):
        new_Q_final[k] = new_Q[k]

    return new_P_final, new_Q_final


# NOTE this function has weird arguments but is for internal use only, so it's probably ok.
# @nb.njit
# def get_monotone_morphing_width(morphing_obj: fu.Morphing) -> fu.Morphing:
#     # TODO change this so that it's a function on the new class

#     longest_dist = 0.0
#     morphing = morphing_obj.morphing_list
#     n = len(morphing)

#     # TODO figure out how to declare the empty list without this
#     res = nbt.List.empty_list(fu.eid_type)

#     k = 0

#     while k < n:
#         event = morphing[k]

#         if event.i_is_vert and event.j_is_vert:
#             res.append(event)
#             k += 1
#             continue

#         elif not event.i_is_vert:
#             new_k = k
#             best_t = event.t

#             while (
#                 new_k < n - 1
#                 and morphing[new_k + 1].i_is_vert == event.i_is_vert
#                 and morphing[new_k + 1].i == event.i
#             ):
#                 new_event = morphing[new_k].copy(morphing_obj.P, morphing_obj.Q)
#                 best_t = max(best_t, new_event.t)
#                 # TODO might be the wrong condition??
#                 if best_t > new_event.t:
#                     new_event.reassign_parameter(best_t, morphing_obj.P, morphing_obj.Q)

#                 longest_dist = max(longest_dist, new_event.dist)
#                 res.append(new_event)

#                 new_k += 1

#             new_event = morphing[new_k].copy(morphing_obj.P, morphing_obj.Q)

#             if best_t > new_event.t:
#                 new_event.reassign_parameter(best_t, morphing_obj.P, morphing_obj.Q)

#             longest_dist = max(longest_dist, new_event.dist)
#             res.append(new_event)
#             k = new_k + 1

#         # TODO might be able to simplify this?
#         elif not event.j_is_vert:
#             new_k = k
#             best_t = event.t

#             while (
#                 new_k < n - 1
#                 and morphing[new_k + 1].j_is_vert == event.j_is_vert
#                 and morphing[new_k + 1].j == event.j
#             ):
#                 new_event = morphing[new_k].copy(morphing_obj.P, morphing_obj.Q)
#                 best_t = max(best_t, new_event.t)

#                 # TODO might be the wrong condition??
#                 if best_t > new_event.t:
#                     new_event.reassign_parameter(best_t, morphing_obj.P, morphing_obj.Q)

#                 longest_dist = max(longest_dist, new_event.dist)
#                 res.append(new_event)

#                 new_k += 1

#             new_event = morphing[new_k].copy(morphing_obj.P, morphing_obj.Q)

#             if best_t > new_event.t:
#                 new_event.reassign_parameter(best_t, morphing_obj.P, morphing_obj.Q)

#             longest_dist = max(longest_dist, new_event.dist)
#             res.append(new_event)
#             k = new_k + 1

#     return fu.Morphing(res, morphing_obj.P, morphing_obj.Q, longest_dist)


@nb.njit
def simplify_polygon_radius(P: np.ndarray, r: float) -> list[int]:
    curr = P[0]
    indices = [0]
    n = P.shape[0]

    for i in range(1, n):
        if np.linalg.norm(P[i] - curr) > r:
            curr = P[i]
            indices.append(i)

    indices.append(n - 1)

    return indices


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

    - `approx` : The output morhing has Frechet distance <= (1+approx)*optimal.

    Importantly, approx can be larger than 1, if you want a really
    rough approximation.
    """
    P_orig = P
    Q_orig = Q

    # Modeled after:
    # https://github.com/sarielhp/retractable_frechet/blob/main/src/frechet.jl#L1686
    frechet_distance = frechet_dist_upper_bound(P, Q)

    # radius of simplification allowed
    r = frechet_distance / (approx_ratio + 4.0)

    while r >= (frechet_distance / (approx_ratio + 4.0)):
        r /= 2.0
        p_indices = simplify_polygon_radius(P, r)
        q_indices = simplify_polygon_radius(Q, r)

        P = np.take(P, p_indices, axis=0)
        Q = np.take(Q, q_indices, axis=0)

        _, _, morphing, frechet_distance, _ = frechet_mono_via_refinement(
            P, Q, (3.0 + approx_ratio) / 4.0
        )

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
