from typing import Optional, Type

import numba as nb
import numpy as np
from numba.experimental import jitclass
from typing_extensions import Self

import frechetlib.frechet_utils as fu
import frechetlib.retractable_frechet as rf


@jitclass
class EventPoint:
    """
    # A vertex edge event descriptor.
    #
    # p: Location of the point being matched. Not necessarily a vetex of
    #    the polygon.
    # i: Vertex number in polygon/or edge number where p lies.
    # type: 1 is for point-vertex, 2 is for point-edge
    # t: Convex combination coefficient if p is on the edge. t=0 means its
    #    the ith vertex, t=1 means it is on the i+1 vertex.
    """

    p: np.ndarray
    i: int
    type: int  # TODO replace this with an enum
    t: Optional[float]

    def __init__(
        self, p_: np.ndarray, i_: int, type_: int, t_: Optional[float]
    ) -> None:
        self.p = p_
        self.i = i_
        self.type = type_
        self.t = t_

    @classmethod
    def make_point_vertex_event(cls: Type[Self], R: np.ndarray, i: int) -> Self:
        return cls(R[i], i, 1, None)

    @classmethod
    def make_point_edge_event(cls: Type[Self], p: np.ndarray, i: int, t: float) -> Self:
        return cls(p, i, 2, t)


def convex_comb(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
    return p * (1.0 - t) + q * t


def events_sequence_make_monotone(
    P: np.ndarray, event_list: list[EventPoint]
) -> list[EventPoint]:
    res = []
    # event_iter = iter(event_list)
    # while (event := next(event_iter, None)) is not None:
    # TODO to make this more efficient, I think some events can be removed? Not totally clear
    # IDEA: I think that the EventPoint struct can be removed and the monotonicity can be
    # enforced by doing two passes on the sequence of EID structs.
    i = 0
    while i < len(event_list):
        if event_list[i].type == 1:  # pt-vertex event
            res.append(event_list[i])
            i += 1
            continue

        j = i
        loc = event_list[i].i
        t = event_list[i].t
        while (
            j < len(event_list)
            and event_list[j + 1].type == 2
            and event_list[j + 1].i == loc
        ):
            t = max(t, event_list[j].t)
            if t > event_list[j].t:
                idx = event_list[j].i
                new_point = convex_comb(P[idx], P[idx + 1], t)
                res.append(EventPoint.make_point_edge_event(new_point, idx, t))
            else:
                res.append(event_list[j])

            j += 1

        i = j

    return res


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


def frechet_mono_via_refinement(P: np.ndarray, Q: np.ndarray, approx: float):
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
    fr_r_mono = 1.0
    fr_retract = 0.0

    while fr_r_mono <= approx * fr_retract:
        retractable_width, _ = rf.retractable_ve_frechet(P, Q)
        # TODO the original code splits the refinement and monotonicity
        # computations, but these can be condensed (simply compute the
        # new refined monotone curve along with the distance, instead of
        # monotonizing, then computing the distance, then getting the new curve)
        monotone_morphing_width = get_monotone_morphing_width(
            P,
        )


# Based on https://github.com/sarielhp/retractable_frechet/blob/main/src/frechet.jl#L155
def get_monotone_morphing_width(morphing: list[rf.EID]) -> float:
    prev_event: rf.EID = morphing[0]
    res = []
    for k in range(1, len(morphing)):
        event = morphing[k]

        # Only happens in vertex-vertex events
        if event.t is None:
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


def simplify_polygon_radius(P: np.ndarray, r: float) -> list[int]:
    # TODO write an actual implementation
    return list(range(len(P)))


def frechet_c_approx(P: np.ndarray, Q: np.ndarray, approx_ratio: float):
    """
    Approximates the continuous Frechet distance between the two input
    curves. Returns a monotone morphing realizing it.

    # Arguments

    - `approx` : The output morhing has Frechet distance <= (1+approx)*optimal.

    Importantly, approx can be larger than 1, if you want a really
    rough approximation.
    """
    # Modeled after:
    # https://github.com/sarielhp/retractable_frechet/blob/main/src/frechet.jl#L1686
    frechet_distance = frechet_dist_upper_bound(P, Q)

    # radius of simplification allowed
    r = frechet_distance / (approx_ratio + 4.0)

    while r >= (frechet_distance / (approx_ratio + 4.0)):
        r /= 2.0
        p_indices = simplify_polygon_radius(P, r)
        q_indices = simplify_polygon_radius(Q, r)

        frechet_distance = frechet_mono_via_refinement(P, Q, (3.0 + approx_ratio) / 4.0)

    # TODO add the stuff about morphing combinations here once I finish the crap above
    return -1
