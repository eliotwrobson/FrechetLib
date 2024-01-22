#! /usr/bin/env python2
# -*- coding: utf-8 -*-

import heapq as hq
from enum import IntEnum
from typing import Self

import numba as nb
import numpy as np
from numba import float64, int32, njit
from numba.experimental import jitclass


class EventType(IntEnum):
    POINT_VERTEX = 1
    POINT_ON_EDGE = 2


@jitclass([("p", float64[:]), ("other_point", float64[:]), ("event_type", int32)])
class EventPoint:
    p: np.ndarray
    i: int
    event_type: EventType
    t: float
    other_point: np.ndarray

    def __init__(
        self,
        p_: np.ndarray,
        i_: int,
        event_type_: EventType,
        t_: float,
        other_point_: np.ndarray,
    ) -> None:
        self.p = p_
        self.i = i_
        self.event_type = event_type_
        self.t = t_
        self.other_point = other_point_


@jitclass
class EID:
    i: int
    i_is_vert: bool
    j: int
    j_is_vert: bool
    dist: float

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

        # Compute the distance
        if self.i_is_vert:
            if self.j_is_vert:
                self.dist = np.linalg.norm(P[self.i] - Q[self.j])

            else:
                self.dist = line_point_distance(Q[self.j], Q[self.j + 1], P[self.i])

        elif self.j_is_vert:
            self.dist = line_point_distance(P[self.i], P[self.i + 1], Q[self.j])
        else:
            raise Exception

    def __lt__(self, other: Self) -> bool:
        return self.dist < other.dist

    def __hash__(self) -> int:
        return hash((self.i, self.i_is_vert, self.j, self.j_is_vert))

    def __eq__(self, other: Self) -> bool:
        return (
            (self.i == other.i)
            and (self.j == other.j)
            and (self.i_is_vert == other.i_is_vert)
            and (self.j_is_vert == other.j_is_vert)
        )


@njit
def line_point_distance(p1: np.ndarray, p2: np.ndarray, q: np.ndarray) -> float:
    """
    Based on: https://stackoverflow.com/a/1501725/2923069
    """
    # Return minimum distance between line segment p1-p2 and point q
    l2 = np.linalg.norm(p1 - p2)  # i.e. |p2-p1|^2 -  avoid a sqrt
    if l2 == 0.0:  # p1 == p2 case
        return np.linalg.norm(q - p1)
    # Consider the line extending the segment, parameterized as v + t (p2 - p1).
    # We find projection of point q onto the line.
    # It falls where t = [(q-p1) . (p2-p1)] / |p2-p1|^2
    # We clamp t from [0,1] to handle points outside the segment vw.
    t = np.dot(q - p1, p2 - p1) / l2

    if t <= 0.0:
        return np.linalg.norm(p1 - q)
    elif t >= 1.0:
        return np.linalg.norm(p2 - q)

    return np.linalg.norm(q - (p1 + t * (p2 - p1)))


@njit
def compute_event_value(P: np.ndarray, Q: np.ndarray, event: EID) -> float:
    if event.i_is_vert:
        if event.j_is_vert:
            return np.linalg.norm(P[event.i] - Q[event.j])

        else:
            return line_point_distance(Q[event.j], Q[event.j + 1], P[event.i])

    elif event.j_is_vert:
        return line_point_distance(P[event.i], P[event.i + 1], Q[event.j])

    raise Exception


@njit
def retractable_frechet(P: np.ndarray, Q: np.ndarray) -> float:
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

    while work_queue:
        curr_event = hq.heappop(work_queue)
        res = max(res, curr_event.dist)
        # print(curr_event.i, " ", curr_event.j, " ", res)

        if curr_event.i == n_p - 1 and curr_event.j == n_q - 1:
            break

        for di, i_vert, dj, j_vert in diffs:
            i = curr_event.i + di
            j = curr_event.j + dj
            next_node = EID(i, i_vert, j, j_vert, P, Q)
            # ev_tuple = (i, i_vert, j, j_vert)

            if (
                i >= n_p
                or j >= n_q
                or (i == n_p and not i_vert)
                or (j == n_q and not j_vert)
                or next_node in seen
            ):
                continue

            seen[next_node] = curr_event
            print(next_node)
            hq.heappush(work_queue, next_node)

    return res
