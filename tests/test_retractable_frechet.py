import time

import numpy as np

import frechetlib.retractable_frechet as rf


def test_classes() -> None:
    rf.EventPoint(
        np.ndarray(2),
        0,
        rf.EventType.POINT_VERTEX,
        0.0,
        np.ndarray(2),
    )


def test_frechet() -> None:
    n = 100
    P = np.random.rand(n, 2)
    Q = np.random.rand(n, 2)
    rf.retractable_frechet(P, Q)

    start = time.perf_counter()
    assert 0.0 != rf.retractable_frechet(P, Q)
    end = time.perf_counter()
    print(end - start)
    assert False
