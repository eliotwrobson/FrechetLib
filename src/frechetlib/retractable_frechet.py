#! /usr/bin/env python2
# -*- coding: utf-8 -*-

from enum import IntEnum

import numpy as np
from numba import float64, int32
from numba.experimental import jitclass

# class syntax


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
