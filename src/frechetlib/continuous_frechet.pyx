# import numpy as np

# import frechetlib.frechet_utils as fu
# import frechetlib.geometry_utils as gu
# import frechetlib.retractable_frechet as rf

from .retractable_frechet import retractable_ve_frechet, retractable_ve_frechet_internal
from .frechet_utils cimport Morphing, add_points_to_make_monotone, frechet_dist_upper_bound, morphing_combine
from .geometry_utils cimport Point, from_curve_indices, numpy_to_point_list
cimport numpy as cnp

cpdef Morphing frechet_mono_via_refinement(
    list P, list Q, float approx
):
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

    cdef Morphing ve_morphing = retractable_ve_frechet_internal(P, Q, None, None, False)

    cdef Morphing monotone_morphing = ve_morphing.copy()
    monotone_morphing.make_monotone()

    # Continue until monotone_morphing.dist <= approx * ve_morphing.dist
    while monotone_morphing.get_dist() > approx * ve_morphing.get_dist():
        # Add points where monotonicity was broken to improve distance
        new_points = add_points_to_make_monotone(ve_morphing)

        # Compute new ve frechet distance for curves
        ve_morphing = retractable_ve_frechet_internal(
            new_points.get_P(),
            new_points.get_Q(),
            None,
            None,
            False
        )

        # Make monotone
        monotone_morphing = ve_morphing.copy()
        monotone_morphing.make_monotone()

        # print("ve: ", ve_morphing.dist)
        # print("monotone: ", monotone_morphing.dist)

    return monotone_morphing

    # np.isclose(
    #     ve_morphing.get_dist(), monotone_morphing.get_dist()
    # )


cdef tuple[list, list] simplify_polygon_radius(list P, float r):
    cdef Point curr = P[0]
    cdef list indices = [0]
    cdef int n = len(P)
    cdef int i, index

    for i in range(1, n):
        if P[i].compute_distance(curr) > r:
            curr = P[i]
            indices.append(i)

    if indices[-1] != n - 1:
        indices.append(n - 1)

    cdef list new_P = []

    for index in indices:
        new_P.append(P[index])

    return new_P, indices


cdef Morphing frechet_c_mono_approx_subcurve(
    list P, list P_subcurve, list p_indices
):
    # TODO add a test case that checks the validity of the Morphing output
    # by this.
    """
    Approximates the Frechet distance between a curve (P) and subcurve
    (P_subcurve). Here, P_subcurve vertices are the vertices of P
    specified by p_indices. That is P_subcurve[i] = P[p_indices[i]].
    """

    cdef list res = []
    cdef float width = 0.0
    cdef int i

    for i in range(len(p_indices) - 1):
        curr_idx = p_indices[i]
        next_idx = p_indices[i + 1]

        next_event = from_curve_indices(
            curr_idx, True, i, True, P, P_subcurve, None, None
        ).get_event()
        width = max(width, next_event.get_dist())
        res.append(next_event)

        for j in range(curr_idx + 1, next_idx):
            next_event = from_curve_indices(
                j, True, i, False, P, P_subcurve, None, None
            ).get_event()
            width = max(width, next_event.get_dist())
            res.append(next_event)

    next_event = from_curve_indices(
        len(P) - 1, True, len(P_subcurve) - 1, True, P, P_subcurve, None, None
    ).get_event()
    width = max(width, next_event.get_dist())
    res.append(next_event)

    return Morphing(res, P, P_subcurve, width)

cdef class FrechetApproxResult:
    def __cinit__(self, double ratio, Morphing morphing):
        self.ratio = ratio
        self.morphing = morphing

    cpdef double get_ratio(self):
        return self.ratio

    cpdef Morphing get_morphing(self):
        return self.morphing

cpdef FrechetApproxResult frechet_c_approx(
    cnp.ndarray P, cnp.ndarray Q, float approx_ratio
):
    """
    Approximates the continuous Frechet distance between the two input
    curves. Returns a monotone morphing realizing it.

    # Arguments

    - `approx` : The output morhing has Frechet distance <= approx*optimal.

    Importantly, approx can be larger than 2, if you want a really
    rough approximation.
    """
    cdef list P_list
    cdef list Q_list

    cdef list P_orig = numpy_to_point_list(P)
    cdef list Q_orig = numpy_to_point_list(Q)
    # print("starting")
    # Modeled after:
    # https://github.com/sarielhp/FrechetDist.jl/blob/main/src/frechet.jl#L810
    cdef float upper_bound_dist = frechet_dist_upper_bound(P, Q)

    # radius of simplification allowed
    cdef float radius = upper_bound_dist / (approx_ratio + 4.0)
    cdef float ratio = approx_ratio + 1.0  # Set to force outer loop to run at least once
    output_morphing = None
    cdef bint should_simplify = True

    while ratio > approx_ratio:
        print("outer", ratio, approx_ratio)

        while should_simplify or radius >= (upper_bound_dist / (approx_ratio + 4.0)):
            print("inner", upper_bound_dist, radius)
            radius /= 2.0
            P_list, p_indices = simplify_polygon_radius(P_orig, radius)
            Q_list, q_indices = simplify_polygon_radius(Q_orig, radius)

            # TODO gotta fix this
            morphing = frechet_mono_via_refinement(P_list, Q_list, (3.0 + approx_ratio) / 4.0)
            # print(morphing.dist, (3.0 + approx_ratio) / 4.0)

            upper_bound_dist = morphing.get_dist()
            should_simplify = False
        print("doen with inner")
        morphing_p = frechet_c_mono_approx_subcurve(P_orig, P_list, p_indices)
        morphing_q = frechet_c_mono_approx_subcurve(Q_orig, Q_list, q_indices)
        print("done with approximation")
        error = max(morphing_p.get_dist(), morphing_q.get_dist())

        morphing_p.make_monotone()
        morphing_q.make_monotone()
        morphing_q.flip()

        assert morphing_p.is_monotone()
        assert morphing_q.is_monotone()
        print("about to combine")
        first_morphing = morphing_combine(morphing, morphing_p)
        first_morphing.make_monotone()
        output_morphing = morphing_combine(morphing_q, first_morphing)
        print("done with morphing combine")
        # TODO I think this morphing will always be monotone?
        assert output_morphing.is_monotone()
        # output_morphing.make_monotone()

        ratio = output_morphing.get_dist() / (upper_bound_dist - 2.0 * error)
        # NOTE This should advance the inner loop on the next iteration.
        should_simplify = True
        # print(ratio, radius, (upper_bound_dist / (approx_ratio + 4.0)))
        print("done with outer")

    if output_morphing is None:
        raise Exception("Output morphing not set!")

    return FrechetApproxResult(ratio, output_morphing)


# def frechet_c_compute(
#     P: np.ndarray, Q: np.ndarray, f_accept_appx: bool = True
# ) -> fu.Morphing:
#     """
#     Compute the exact continuous (monotone) Frechet distance between the
#     two polygons. It should be reasonably fast.

#     This function is somewhat slower than the approximate versions. Use it
#     only if you really want the exact answer. Consider using
#     frechet_continous_approx instead.

#     # Details

#     This works by first computing a very rough approximation, followed by
#     distance senstiave simplification of the curves. It then compute the
#     monotone fr_ve_r distance between the simplified curves, and it
#     combine it to get a distance between the two original cuves. It makre
#     sure the answers are the same, otherwise, it repeates with a finer
#     simplification/approximation till they are equal.

#     Finally, the algorithm uses the fr_ve_r_with_offests distance between
#     the two simplified curves to comptue a lower bound, and make sure this
#     is equal to the Frechet distance computed. If they are equal, then the
#     upper/lower bounds on the Frechet distance of the two curves are the
#     same, which implies that the computed distance is indeed the desired
#     Frechet distance.

#     # More details

#     To really ensure converges, the monotone distance computed between the
#     simplification is computed using refinement, so tha the ve_r distance
#     """

#     baseline_ratio, baseline_morphing = frechet_c_approx(P, Q, 2.0)
#     approx_refinement = 1.001

#     min_approx_ratio = min(
#         1.0 + (P.shape[0] + Q.shape[0]) / (100.0 * baseline_morphing.get_dist()), 1.1
#     )

#     # If initial ratio is good enough, use this morphing
#     if baseline_ratio <= min_approx_ratio:
#         morphing = baseline_morphing
#         ratio = baseline_ratio

#     # Otherwise recompute
#     else:
#         ratio, morphing = frechet_c_approx(P, Q, min_approx_ratio)

#     Pl, Ql = morphing.extract_vertex_radii()
#     lower_bound = morphing.get_dist() / ratio

#     factor = 4.0
#     while True:
#         # print("**** Exact computation inner loop starting ****")
#         # Vectorized sums
#         Pz = (lower_bound - Pl) / factor
#         Qz = (lower_bound - Ql) / factor

#         Ps = fu.simplify_polygon_radii(P, Pz)
#         Qs = fu.simplify_polygon_radii(Q, Qz)

#         # TODO need to refactor to reduce the distance to simplified versions of the
#         # curves (i.e. using offsets when defining event values, or something like that)
#         # see https://github.com/sarielhp/FrechetDist.jl/blob/main/src/frechet.jl#L103
#         mid_morphing, is_exact = frechet_mono_via_refinement(Ps, Qs, approx_refinement)

#         # The P and Q from the refinement without offsets
#         morphing_P = rf.retractable_ve_frechet(
#             P, mid_morphing.get_P(), None, None, False
#         )
#         morphing_Q = rf.retractable_ve_frechet(
#             mid_morphing.get_Q(), Q, None, None, False
#         )

#         # NOTE This apparently does not require flipping? Testing with large input
#         # it works, and my tests are still passing.
#         morphing_P.make_monotone()
#         morphing_Q.make_monotone()
#         # morphing_Q.flip()

#         # print("Mid morphing end PRM", mid_morphing.get_prm())
#         # print("P morphing end PRM", morphing_P.get_prm())
#         # print("Q morphing end PRM", morphing_Q.get_prm())

#         # Do the combination
#         # print("About to combine in inner loop")
#         first_morphing = fu.morphing_combine(mid_morphing, morphing_P)
#         first_morphing.make_monotone()
#         # print("First combined morphing end PRM", first_morphing.get_prm())
#         # print("Q morphing end PRM", morphing_Q.get_prm())
#         combined_morphing = fu.morphing_combine(morphing_Q, first_morphing)

#         # Try shooting for the opt?
#         if np.isclose(combined_morphing.get_dist(), mid_morphing.get_dist()):
#             factor *= 2.0

#         # TODO Triple check these are the correct offsets
#         _, morphing_offsets_P = morphing_P.extract_vertex_radii()
#         morphing_offsets_Q, _ = morphing_Q.extract_vertex_radii()

#         morphing_with_offsets = rf.retractable_ve_frechet(
#             mid_morphing.get_P(),
#             mid_morphing.get_Q(),
#             morphing_offsets_P,
#             morphing_offsets_Q,
#             False,
#         )

#         # If distances are equal, return the simpler of the two
#         # (computed without offsets)
#         # print("offset dist:", morphing_with_offsets.dist)
#         if np.isclose(morphing_with_offsets.get_dist(), combined_morphing.get_dist()):
#             return combined_morphing

#         factor *= 2.0
#         approx_refinement = (1.0 + approx_refinement) / 2.0

#         if f_accept_appx and (
#             1.000001 * combined_morphing.get_dist() < morphing_with_offsets.get_dist()
#         ):
#             # print("HERE")
#             # print(combined_morphing.dist, morphing_with_offsets.dist)
#             # assert False
#             return combined_morphing
