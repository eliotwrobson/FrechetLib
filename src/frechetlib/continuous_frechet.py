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


def frechet_mono_via_refinement(
    P: np.ndarray, Q: np.ndarray, approx: float
) -> tuple[np.ndarray, np.ndarray, list[fu.EID], float, bool]:
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

    # Set these so the loop runs at least once
    fr_r_mono = -1.0
    fr_retract = 0.0
    f_exact = False

    while fr_r_mono <= approx * fr_retract:
        fr_retract, ve_morphing = rf.retractable_ve_frechet(P, Q)

        fr_r_mono, monotone_morphing = get_monotone_morphing_width(
            nbt.List(ve_morphing)
        )

        if np.isclose(fr_retract, fr_r_mono):
            f_exact = True
            break
        elif fr_r_mono <= approx * fr_retract:
            break

        P, Q = add_points_to_make_monotone()

    return P, Q, monotone_morphing, fr_r_mono, f_exact


def add_points_to_make_monotone(
    P: np.ndarray,
    Q: np.ndarray,
    morphing: list[fu.EID],
) -> tuple[np.ndarray, np.ndarray]:
    p_points, q_points = _add_points_to_make_monotone(P, Q, nbt.List(morphing))
    return np.array(p_points), np.array(q_points)


@nb.njit
def _add_points_to_make_monotone(
    P: np.ndarray,
    Q: np.ndarray,
    morphing: nbt.List[fu.EID],
) -> tuple[float, list[fu.EID]]:
    # TODO add intermediate vertices here
    # Doing the same here:
    # https://github.com/sarielhp/FrechetDist.jl/blob/main/src/frechet.jl#L626

    # First, add points to P
    new_P = []
    k = 0
    while k < len(morphing):
        # Vertex-vertex event, can ignore
        if morphing[k].i_is_vert:
            new_P.append(P[morphing[k].i])
            k += 1
            continue

        new_k = k
        loc = morphing[k].i
        offsets = []

        while (
            new_k < len(morphing) - 1
            and not morphing[new_k + 1].i_is_vert
            and morphing[new_k + 1].i == loc
        ):
            offsets.append(morphing[new_k].t)
            new_k += 1

        # [k:new_k) is the indices of points that are on the same segment
        monotone = True
        offsets = sorted(offsets)
        for j in range(len(offsets) - 1):
            monotone = monotone and (offsets[j] <= offsets[j + 1])

        for j in range(len(offsets)):
            new_P.append(fu.convex_comb(P[loc], P[loc + 1], offsets[j]))
            if not monotone and j < len(offsets) - 1:
                new_P.append(
                    fu.convex_comb(
                        P[loc], P[loc + 1], (offsets[j] + offsets[j + 1]) / 2
                    )
                )
        # TODO I think there's an off by one here
        k = new_k + 1

    # Next, add points to Q, same as above but hard to share logic
    new_Q = []
    k = 0
    while k < len(morphing):
        # Vertex-vertex event, can ignore
        if morphing[k].j_is_vert:
            new_Q.append(Q[morphing[k].j])
            k += 1
            continue

        new_k = k
        loc = morphing[k].i
        offsets = []

        while (
            new_k < len(morphing) - 1
            and not morphing[new_k + 1].j_is_vert
            and morphing[new_k + 1].j == loc
        ):
            offsets.append(morphing[new_k].t)
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

    return new_P, new_Q


# Based on https://github.com/sarielhp/retractable_frechet/blob/main/src/frechet.jl#L155
# NOTE this function has weird arguments but is for internal use only, so it's probably ok.
@nb.njit
def get_monotone_morphing_width(
    morphing: nbt.List[fu.EID],
) -> tuple[float, list[fu.EID]]:
    prev_event: fu.EID = morphing[0]
    res = []
    longest_dist = prev_event.dist

    for k in range(1, len(morphing)):
        event = morphing[k]

        # Only happens in vertex-vertex events
        if event.i_is_vert and event.i_is_vert:
            res.append(prev_event)
            prev_event = event

            continue

        # Monotonicity case for when i or j stays vertex and the other varies, but the
        # coefficient goes down.
        if (prev_event.i_is_vert == event.i_is_vert and prev_event.i == event.i) or (
            prev_event.j_is_vert == event.j_is_vert and prev_event.j == event.j
        ):
            if prev_event.t > event.t:
                prev_event = event
        else:
            res.append(prev_event)
            prev_event = event

    res.append(prev_event)

    return longest_dist, res


@nb.njit
def simplify_polygon_radius(P: np.ndarray, r: float) -> list[int]:
    curr = P[0]
    indices = [0]

    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - curr) > r:
            curr = P[i]
            indices.append(i)

    return indices


@nb.njit
def frechet_c_mono_approx_subcurve(
    P: np.ndarray, P_subcurve: np.ndarray, p_indices: list[int]
) -> list[fu.EID]:
    """
    Approximates the Frechet distance between a curve (P) and subcurve
    (P_subcurve). Here, P_subcurve vertices are the vertices of P
    specified by p_indices. That is P_subcurve[i] = P[p_indices[i]].
    """

    res = []
    for i in range(len(p_indices) - 1):
        curr_idx = p_indices[i]
        next_idx = p_indices[i + 1]

        res.append(fu.EID(i, True, curr_idx, True, P, P_subcurve))

        for j in range(curr_idx + 1, next_idx):
            res.append(fu.EID(i, True, j, False, P, P_subcurve))

    res.append(fu.EID(len(P) - 1, True, len(P_subcurve) - 1, True, P, P_subcurve))

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

    morphing_p = frechet_c_mono_approx_subcurve(P_orig, P, p_indices)
    morphing_q = frechet_c_mono_approx_subcurve(Q_orig, Q, q_indices)

    first_combined = fu.morphing_combine(P_orig, P, Q, morphing_p, morphing)
    _, first_combined_monotone = get_monotone_morphing_width(nbt.List(first_combined))
    width, final_combined = fu.morphing_combine(
        P_orig, Q, Q_orig, first_combined_monotone, morphing_q
    )
    # TODO might be need to run through this a second time in a loop? Not sure
    return final_combined
