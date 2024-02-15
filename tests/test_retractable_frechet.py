import time

import numpy as np
from utils import leq_with_tolerance

import frechetlib.retractable_frechet as rf


def test_retractable_frechet() -> None:
    n = 50
    # Just to make the test case always generate the same number
    noise_limit = 2.0
    d = 2
    np.random.seed(12345)
    P = np.random.uniform(low=noise_limit, high=100.0, size=(n, d))
    Q = P + np.random.uniform(low=0.0, high=noise_limit, size=(n, d))

    frechet_limit = np.linalg.norm(np.ones(d) * noise_limit)

    rf.retractable_ve_frechet(P, Q)
    start = time.perf_counter()
    frechet_dist, morphing = rf.retractable_ve_frechet(P, Q)
    end = time.perf_counter()

    print(end - start)

    assert frechet_dist <= frechet_limit
    assert len(morphing) == n * 2

    for event in morphing:
        assert leq_with_tolerance(frechet_dist, event.dist)

    assert morphing[0].i == 0
    assert morphing[0].i_is_vert
    assert morphing[0].j == 0
    assert morphing[0].j_is_vert
    assert morphing[0].t == 0.0

    # Check that only one vertex changes at a time,
    # ignoring the first event.
    for k in range(1, len(morphing) - 1):
        assert (morphing[k].i == morphing[k + 1].i) ^ (
            morphing[k].j == morphing[k + 1].j
        )
