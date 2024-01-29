#! /usr/bin/env python2
# -*- coding: utf-8 -*-

import heapq as hq
from typing import Optional

import frechet_utils as fu
import numba as nb
import numpy as np
from numba.experimental import jitclass
from typing_extensions import Self


@jitclass
class EID:
    i: int
    i_is_vert: bool
    j: int
    j_is_vert: bool
    dist: float
    t: Optional[float]

    def __init__(
        self,
        i_: int,
        i_is_vert_: bool,
        j_: int,
        j_is_vert_: bool,
        P: np.ndarray,
        Q: np.ndarray,
    ) -> None:
        self.i = i_
        self.i_is_vert = i_is_vert_
        self.j = j_
        self.j_is_vert = j_is_vert_
        self.t = None

        # Compute the distance
        if self.i_is_vert:
            if self.j_is_vert:
                self.dist = float(np.linalg.norm(P[self.i] - Q[self.j]))

            else:
                self.dist, self.t = fu.line_point_distance(
                    Q[self.j], Q[self.j + 1], P[self.i]
                )

        elif self.j_is_vert:
            self.dist, self.t = fu.line_point_distance(
                P[self.i], P[self.i + 1], Q[self.j]
            )
        else:
            raise Exception

    def __lt__(self, other: Self) -> bool:
        return self.dist < other.dist

    def __hash__(self) -> int:
        return hash((self.i, self.i_is_vert, self.j, self.j_is_vert))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EID):
            return False

        return (
            (self.i == other.i)
            and (self.j == other.j)
            and (self.i_is_vert == other.i_is_vert)
            and (self.j_is_vert == other.j_is_vert)
        )


@nb.njit
def retractable_ve_frechet(P: np.ndarray, Q: np.ndarray) -> tuple[float, list[EID]]:
    start_node = EID(0, True, 0, True, P, Q)
    start_node_1 = EID(0, False, 0, True, P, Q)
    start_node_2 = EID(0, True, 0, False, P, Q)
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

        for di, i_vert, dj, j_vert in diffs:
            i = curr_event.i + di
            j = curr_event.j + dj
            next_node = EID(i, i_vert, j, j_vert, P, Q)

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

    morphing = [last_event]

    while last_event in seen:
        last_event = seen[last_event]
        morphing.append(last_event)

    # TODO maybe add final event??
    morphing.reverse()

    return res, morphing
