import numba.typed as nbt
import numpy as np
from numba import njit  # type: ignore[attr-defined]

import frechetlib.frechet_utils as fu
import frechetlib.retractable_frechet as rf


@njit
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

    ve_morphing = rf.retractable_ve_frechet(P, Q, None, None, False)

    monotone_morphing = ve_morphing.copy()
    monotone_morphing.make_monotone()

    # Continue until monotone_morphing.dist <= approx * ve_morphing.dist
    while monotone_morphing.dist > approx * ve_morphing.dist:
        # Add points where monotonicity was broken to improve distance
        new_P, new_Q = fu.add_points_to_make_monotone(ve_morphing)

        # Compute new ve frechet distance for curves
        ve_morphing = rf.retractable_ve_frechet(new_P, new_Q, None, None, False)

        # Make monotone
        monotone_morphing = ve_morphing.copy()
        monotone_morphing.make_monotone()

        # print("ve: ", ve_morphing.dist)
        # print("monotone: ", monotone_morphing.dist)

    return monotone_morphing, np.isclose(ve_morphing.dist, monotone_morphing.dist)


@njit
def simplify_polygon_radius(P: np.ndarray, r: float) -> tuple[np.ndarray, list[int]]:
    curr = P[0]
    indices = [0]
    n = P.shape[0]

    for i in range(1, n):
        if np.linalg.norm(P[i] - curr) > r:
            curr = P[i]
            indices.append(i)

    if indices[-1] != n - 1:
        indices.append(n - 1)

    new_P = np.empty((len(indices), P.shape[1]))

    for k in range(len(indices)):
        new_P[k] = P[indices[k]]

    return new_P, indices


@njit
def frechet_c_mono_approx_subcurve(
    P: np.ndarray, P_subcurve: np.ndarray, p_indices: list[int]
) -> fu.Morphing:
    # TODO add a test case that checks the validity of the Morphing output
    # by this.
    """
    Approximates the Frechet distance between a curve (P) and subcurve
    (P_subcurve). Here, P_subcurve vertices are the vertices of P
    specified by p_indices. That is P_subcurve[i] = P[p_indices[i]].
    """

    res = nbt.List.empty_list(fu.eid_type)
    width = 0.0
    for i in range(len(p_indices) - 1):
        curr_idx = p_indices[i]
        next_idx = p_indices[i + 1]

        _, next_event = fu.from_curve_indices(
            curr_idx, True, i, True, P, P_subcurve, None, None
        )
        width = max(width, next_event.dist)
        res.append(next_event)

        for j in range(curr_idx + 1, next_idx):
            # print("index", i, j)
            _, next_event = fu.from_curve_indices(
                j, True, i, False, P, P_subcurve, None, None
            )
            width = max(width, next_event.dist)
            res.append(next_event)

    _, next_event = fu.from_curve_indices(
        len(P) - 1, True, len(P_subcurve) - 1, True, P, P_subcurve, None, None
    )
    width = max(width, next_event.dist)
    res.append(next_event)

    return fu.Morphing(res, P, P_subcurve, width)


@njit
def frechet_c_approx(
    P: np.ndarray, Q: np.ndarray, approx_ratio: float
) -> tuple[float, fu.Morphing]:
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
    # print("starting")
    # Modeled after:
    # https://github.com/sarielhp/FrechetDist.jl/blob/main/src/frechet.jl#L810
    upper_bound_dist = fu.frechet_dist_upper_bound(P, Q)

    # radius of simplification allowed
    radius = upper_bound_dist / (approx_ratio + 4.0)
    ratio = approx_ratio + 1.0  # Set to force outer loop to run at least once
    output_morphing = None
    should_simplify = True

    while ratio > approx_ratio:
        print("outer", ratio, approx_ratio)

        while should_simplify or radius >= (upper_bound_dist / (approx_ratio + 4.0)):
            print("inner", upper_bound_dist, radius)
            radius /= 2.0
            P, p_indices = simplify_polygon_radius(P_orig, radius)
            Q, q_indices = simplify_polygon_radius(Q_orig, radius)

            morphing, _ = frechet_mono_via_refinement(P, Q, (3.0 + approx_ratio) / 4.0)
            # print(morphing.dist, (3.0 + approx_ratio) / 4.0)

            upper_bound_dist = morphing.dist
            should_simplify = False

        morphing_p = frechet_c_mono_approx_subcurve(P_orig, P, p_indices)
        morphing_q = frechet_c_mono_approx_subcurve(Q_orig, Q, q_indices)

        error = max(morphing_p.dist, morphing_q.dist)

        morphing_p.make_monotone()
        morphing_q.make_monotone()
        morphing_q.flip()

        assert morphing_p.is_monotone()
        assert morphing_q.is_monotone()

        first_morphing = fu.morphing_combine(morphing, morphing_p)
        first_morphing.make_monotone()
        output_morphing = fu.morphing_combine(morphing_q, first_morphing)

        # TODO I think this morphing will always be monotone?
        assert output_morphing.is_monotone()
        # output_morphing.make_monotone()

        ratio = output_morphing.dist / (upper_bound_dist - 2.0 * error)
        # NOTE This should advance the inner loop on the next iteration.
        should_simplify = True
        print(ratio, radius, (upper_bound_dist / (approx_ratio + 4.0)))

    if output_morphing is None:
        raise Exception("Output morphing not set!")

    return ratio, output_morphing


# @njit
def frechet_c_compute(
    P: np.ndarray, Q: np.ndarray, f_accept_appx: bool = True
) -> fu.Morphing:
    """
    Compute the exact continuous (monotone) Frechet distance between the
    two polygons. It should be reasonably fast.

    This function is somewhat slower than the approximate versions. Use it
    only if you really want the exact answer. Consider using
    frechet_continous_approx instead.

    # Details

    This works by first computing a very rough approximation, followed by
    distance senstiave simplification of the curves. It then compute the
    monotone fr_ve_r distance between the simplified curves, and it
    combine it to get a distance between the two original cuves. It makre
    sure the answers are the same, otherwise, it repeates with a finer
    simplification/approximation till they are equal.

    Finally, the algorithm uses the fr_ve_r_with_offests distance between
    the two simplified curves to comptue a lower bound, and make sure this
    is equal to the Frechet distance computed. If they are equal, then the
    upper/lower bounds on the Frechet distance of the two curves are the
    same, which implies that the computed distance is indeed the desired
    Frechet distance.

    # More details

    To really ensure converges, the monotone distance computed between the
    simplification is computed using refinement, so tha the ve_r distance
    """

    baseline_ratio, baseline_morphing = frechet_c_approx(P, Q, 2.0)
    approx_refinement = 1.001

    min_approx_ratio = min(
        1.0 + (P.shape[0] + Q.shape[0]) / (100.0 * baseline_morphing.dist), 1.1
    )

    # If initial ratio is good enough, use this morphing
    if baseline_ratio <= min_approx_ratio:
        morphing = baseline_morphing
        ratio = baseline_ratio

    # Otherwise recompute
    else:
        ratio, morphing = frechet_c_approx(P, Q, min_approx_ratio)

    Pl, Ql = morphing.extract_vertex_radii()
    lower_bound = morphing.dist / ratio

    factor = 4.0
    while True:
        print("**** Exact computation inner loop starting ****")
        # Vectorized sums
        Pz = (lower_bound - Pl) / factor
        Qz = (lower_bound - Ql) / factor

        Ps = fu.simplify_polygon_radii(P, Pz)
        Qs = fu.simplify_polygon_radii(Q, Qz)

        # TODO need to refactor to reduce the distance to simplified versions of the
        # curves (i.e. using offsets when defining event values, or something like that)
        # see https://github.com/sarielhp/FrechetDist.jl/blob/main/src/frechet.jl#L103
        mid_morphing, is_exact = frechet_mono_via_refinement(Ps, Qs, approx_refinement)

        # The P and Q from the refinement without offsets
        morphing_P = rf.retractable_ve_frechet(P, mid_morphing.P, None, None, False)
        morphing_Q = rf.retractable_ve_frechet(mid_morphing.Q, Q, None, None, False)

        # NOTE I think the flipping here is a little bit messed up. This is probably where
        # the bug is happening.
        morphing_P.make_monotone()
        morphing_Q.make_monotone()
        # morphing_Q.flip()

        print("Mid morphing end PRM", mid_morphing.get_prm())
        print("P morphing end PRM", morphing_P.get_prm())
        # print("Q morphing end PRM", morphing_Q.get_prm())

        # Do the combination
        print("About to combine in inner loop")
        first_morphing = fu.morphing_combine(mid_morphing, morphing_P)
        first_morphing.make_monotone()
        print("First combined morphing end PRM", first_morphing.get_prm())
        print("Q morphing end PRM", morphing_Q.get_prm())
        combined_morphing = fu.morphing_combine(morphing_Q, first_morphing)

        # Try shooting for the opt?
        if np.isclose(combined_morphing.dist, mid_morphing.dist):
            factor *= 2.0

        # TODO Triple check these are the correct offsets
        _, morphing_offsets_P = morphing_P.extract_vertex_radii()
        morphing_offsets_Q, _ = morphing_Q.extract_vertex_radii()

        morphing_with_offsets = rf.retractable_ve_frechet(
            mid_morphing.P,
            mid_morphing.Q,
            morphing_offsets_P,
            morphing_offsets_Q,
            False,
        )

        # If distances are equal, return the simpler of the two
        # (computed without offsets)
        # print("offset dist:", morphing_with_offsets.dist)
        if np.isclose(morphing_with_offsets.dist, combined_morphing.dist):
            return combined_morphing

        factor *= 2.0
        approx_refinement = (1.0 + approx_refinement) / 2.0

        # TODO I flipped the inequality here from the original Julia code.
        # make sure this is correct (though I think it is).
        if f_accept_appx and (
            1.000001 * combined_morphing.dist < morphing_with_offsets.dist
        ):
            # print("HERE")
            # print(combined_morphing.dist, morphing_with_offsets.dist)
            # assert False
            return combined_morphing
