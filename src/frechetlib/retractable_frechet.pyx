import heapq as hq
import typing as t

import numpy as np

from .frechet_utils cimport Morphing
from .geometry_utils cimport from_curve_indices


cpdef Morphing retractable_ve_frechet(
    cnp.ndarray P,
    cnp.ndarray Q,
    cnp.ndarray P_offs,
    cnp.ndarray Q_offs,
    bint summed,
):
    start_node = from_curve_indices(
        0, True, 0, True, P, Q, P_offs, Q_offs
    ).get_event()

    first_event = from_curve_indices(0, False, 0, True, P, Q, P_offs, Q_offs)
    start_tuple_1 = (first_event.get_heap_key(), first_event.get_event())

    second_event = from_curve_indices(0, True, 0, False, P, Q, P_offs, Q_offs)
    start_tuple_2 = (second_event.get_heap_key(), second_event.get_event())
    cdef list work_queue = [start_tuple_1, start_tuple_2]

    cdef dict seen = {start_tuple_1[1]: start_node, start_tuple_2[1]: start_node}
    hq.heapify(work_queue)

    cdef int n_p = P.shape[0]
    cdef int n_q = Q.shape[0]
    diffs = ((1, True, 0, False), (0, False, 1, True))
    last_event = start_node

    while work_queue:
        curr_cost, curr_event = hq.heappop(work_queue)

        if curr_event.get_i() == n_p - 1 and curr_event.get_j() == n_q - 1:
            last_event = curr_event
            break

        for di, i_vert, dj, j_vert in diffs:
            # Start with bounds creation and checking
            i = curr_event.get_i() + di
            j = curr_event.get_j() + dj

            if i >= n_p or j >= n_q:
                continue

            next_event = from_curve_indices(
                i, i_vert, j, j_vert, P, Q, P_offs, Q_offs
            )

            next_cost, next_node = next_event.get_heap_key(), next_event.get_event()

            # NOTE in the bottleneck Frechet case, we always take the
            # local optimum, but in the summed case, we need to add
            if summed:
                next_cost += curr_cost

            next_tuple = (next_cost, next_node)

            if next_node in seen:
                continue

            seen[next_node] = curr_event
            hq.heappush(work_queue, next_tuple)

    morphing = [last_event]

    cdef double res = last_event.get_dist()
    while last_event in seen:
        last_event = seen[last_event]

        if summed:
            res += last_event.get_dist()
        else:
            res = max(res, last_event.get_dist())

        morphing.append(last_event)

    assert last_event == start_node
    # TODO maybe add final event??
    morphing.reverse()

    return Morphing(morphing, P, Q, res)
