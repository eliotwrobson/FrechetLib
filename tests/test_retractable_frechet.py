import time

import numba.typed as nbt
import numpy as np
from utils import leq_with_tolerance

import frechetlib.frechet_utils as fu
import frechetlib.retractable_frechet as rf


def test_retractable_frechet() -> None:
    n = 5000
    # Just to make the test case always generate the same number
    noise_limit = 5.0
    d = 2
    np.random.seed(12345)
    P = np.random.uniform(low=noise_limit, high=100.0, size=(n, d))
    Q = P + np.random.uniform(low=-noise_limit, high=noise_limit, size=(n, d))

    rf.retractable_ve_frechet(P, Q)
    start = time.perf_counter()
    morphing = rf.retractable_ve_frechet(P, Q)
    end = time.perf_counter()

    print(end - start)

    # TODO fix this limit assertion, not sure it works right.

    # frechet_limit = np.linalg.norm(np.ones(d) * 2 * noise_limit)
    # assert frechet_dist <= frechet_limit
    assert len(morphing.morphing_list) == n * 2

    for event in morphing.morphing_list:
        assert leq_with_tolerance(morphing.dist, event.dist)

    assert morphing.morphing_list[0].i == 0
    assert morphing.morphing_list[0].i_is_vert
    assert morphing.morphing_list[0].j == 0
    assert morphing.morphing_list[0].j_is_vert
    assert morphing.morphing_list[0].t == 0.0

    # Check that only one vertex changes at a time,
    # ignoring the first event.
    for k in range(1, len(morphing.morphing_list) - 1):
        assert (morphing.morphing_list[k].i == morphing.morphing_list[k + 1].i) ^ (
            morphing.morphing_list[k].j == morphing.morphing_list[k + 1].j
        )
