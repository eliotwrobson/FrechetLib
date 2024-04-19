import time

import frechetlib.frechet_utils as fu
import frechetlib.retractable_frechet as rf
import numpy as np
from utils import leq_with_tolerance


def test_retractable_basic() -> None:
    P = np.array([[1.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.0, 1.0]])

    bottleneck_morphing = rf.retractable_ve_frechet(P, Q, None, None, False)
    assert len(bottleneck_morphing.morphing_list) == 4
    assert np.isclose(bottleneck_morphing.dist, 1.0)

    # TODO there might be an extra event here? Either way, this is being done
    # consistently so curves with the same number of vertices should be comparable
    summed_morphing = rf.retractable_ve_frechet(P, Q, None, None, True)
    assert len(summed_morphing.morphing_list) == 4
    assert np.isclose(summed_morphing.dist, 4.0)

    P = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.0, 1.0]])

    summed_morphing = rf.retractable_ve_frechet(P, Q, None, None, True)
    assert len(summed_morphing.morphing_list) == 5
    assert np.isclose(summed_morphing.dist, 5.0)


def test_retractable_weird() -> None:
    P = np.array(
        [
            [0.0, 0.0],
            [0.15, 0.15],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6],
            [0.7, 0.7],
            [0.85, 0.85],
            [1.0, 1.0],
        ]
    )

    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])
    morphing = rf.retractable_ve_frechet(P, Q, None, None, False)
    assert np.isclose(morphing.dist, 0.14142135623730948)
    morphing.make_monotone()

    assert np.isclose(morphing.dist, 0.14142135623730948)


def test_retractable_frechet() -> None:
    n = 50
    # Just to make the test case always generate the same number
    noise_limit = 5.0
    d = 2
    np.random.seed(12345)
    P = np.random.uniform(low=noise_limit, high=100.0, size=(n, d))
    Q = P + np.random.uniform(low=-noise_limit, high=noise_limit, size=(n, d))

    rf.retractable_ve_frechet(P, Q, None, None, False)
    start = time.perf_counter()
    morphing = rf.retractable_ve_frechet(P, Q, None, None, False)
    end = time.perf_counter()

    print(end - start)

    # TODO fix this limit assertion, not sure it works right.
    # frechet_limit = np.linalg.norm(np.ones(d) * 2 * noise_limit)
    # assert frechet_dist <= frechet_limit

    fu._print_event_list(morphing)

    assert len(morphing.morphing_list) == n * 2

    for event in morphing.morphing_list:
        assert leq_with_tolerance(morphing.dist, event.dist)

    assert morphing.morphing_list[0].i == 0
    assert morphing.morphing_list[0].i_is_vert
    assert morphing.morphing_list[0].j == 0
    assert morphing.morphing_list[0].j_is_vert
    assert morphing.morphing_list[0].t_i == 0.0
    assert morphing.morphing_list[0].t_j == 0.0

    # Check that only one vertex changes at a time,
    # ignoring the first event.
    for k in range(1, len(morphing.morphing_list) - 1):
        assert (morphing.morphing_list[k].i == morphing.morphing_list[k + 1].i) ^ (
            morphing.morphing_list[k].j == morphing.morphing_list[k + 1].j
        )
