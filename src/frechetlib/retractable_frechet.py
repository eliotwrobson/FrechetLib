#! /usr/bin/env python2
# -*- coding: utf-8 -*-

import heapq as hq
from typing import Optional

import numba as nb
import numba.typed as nbt
import numpy as np

import frechetlib.frechet_utils as fu


@nb.njit
def retractable_ve_frechet(P: np.ndarray, Q: np.ndarray) -> fu.Morphing:
    start_node = fu.EID(0, True, 0, True, P, Q)
    start_node_1 = fu.EID(0, False, 0, True, P, Q)
    start_node_2 = fu.EID(0, True, 0, False, P, Q)
    work_queue = [start_node_1, start_node_2]

    seen = {start_node_1: start_node, start_node_2: start_node}
    hq.heapify(work_queue)

    n_p = P.shape[0]
    n_q = Q.shape[0]
    diffs = ((1, True, 0, False), (0, False, 1, True))
    res = start_node.dist
    last_event = start_node

    while work_queue:
        curr_event = hq.heappop(work_queue)
        res = max(res, curr_event.dist)

        if curr_event.i == n_p - 1 and curr_event.j == n_q - 1:
            last_event = curr_event
            break

        for di, i_vert, dj, j_vert in diffs:
            i = curr_event.i + di
            j = curr_event.j + dj
            next_node = fu.EID(i, i_vert, j, j_vert, P, Q)

            if (
                i >= n_p
                or j >= n_q
                or (i == n_p and not i_vert)
                or (j == n_q and not j_vert)
                or next_node in seen
            ):
                continue

            seen[next_node] = curr_event
            hq.heappush(work_queue, next_node)

    morphing = nbt.List([last_event])

    while last_event in seen:
        last_event = seen[last_event]
        morphing.append(last_event)

    # TODO maybe add final event??
    morphing.reverse()

    return fu.Morphing(morphing, P, Q, res)
