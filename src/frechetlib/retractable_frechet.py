#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import heapq as hq
import typing as t

import numba.typed as nbt
import numpy as np
from numba import (  # type: ignore[attr-defined]
    boolean,
    float64,
    njit,
    optional,
)

import frechetlib.frechet_utils as fu


@njit(
    fu.Morphing.class_type.instance_type(  # type: ignore[attr-defined]
        float64[:, :],
        float64[:, :],
        optional(float64[:]),
        optional(float64[:]),
        boolean,
    )
)
# TODO this doesn't work, fix it in other places.
def retractable_ve_frechet(
    P: np.ndarray,
    Q: np.ndarray,
    P_offs: t.Optional[np.ndarray],
    Q_offs: t.Optional[np.ndarray],
    summed: bool,
) -> fu.Morphing:
    _, start_node = fu.from_curve_indices(0, True, 0, True, P, Q, P_offs, Q_offs)
    start_tuple_1 = fu.from_curve_indices(0, False, 0, True, P, Q, P_offs, Q_offs)
    start_tuple_2 = fu.from_curve_indices(0, True, 0, False, P, Q, P_offs, Q_offs)
    work_queue = [start_tuple_1, start_tuple_2]

    seen = {start_tuple_1[1]: start_node, start_tuple_2[1]: start_node}
    hq.heapify(work_queue)

    n_p = P.shape[0]
    n_q = Q.shape[0]
    diffs = ((1, True, 0, False), (0, False, 1, True))
    last_event = start_node

    while work_queue:
        curr_cost, curr_event = hq.heappop(work_queue)

        if curr_event.i == n_p - 1 and curr_event.j == n_q - 1:
            last_event = curr_event
            break

        for di, i_vert, dj, j_vert in diffs:
            # Start with bounds creation and checking
            i = curr_event.i + di
            j = curr_event.j + dj

            if i >= n_p or j >= n_q:
                continue

            next_cost, next_node = fu.from_curve_indices(
                i, i_vert, j, j_vert, P, Q, P_offs, Q_offs
            )

            # NOTE in the bottleneck Frechet case, we always take the
            # local optimum, but in the summed case, we need to
            if summed:
                next_cost += curr_cost

            next_tuple = (next_cost, next_node)

            if next_node in seen:
                continue

            seen[next_node] = curr_event
            hq.heappush(work_queue, next_tuple)

    morphing = nbt.List([last_event])

    res = last_event.dist
    while last_event in seen:
        last_event = seen[last_event]

        if summed:
            res += last_event.dist
        else:
            res = max(res, last_event.dist)

        morphing.append(last_event)

    # TODO maybe add final event??
    morphing.reverse()

    return fu.Morphing(morphing, P, Q, res)
