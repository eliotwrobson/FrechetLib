import copy

import numpy as np
import utils as u

import frechetlib.frechet_utils as fu


def test_convex_comb() -> None:
    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0, 1.0])
    q = np.array([1.0, 1.0])
    dist, t, point = fu.line_point_distance(p1, p2, q)
    assert 0.0 <= t <= 1.0


def test_eid_copy() -> None:
    # Contents here are not really important for this test
    P = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    Q = np.array([[0.0, 1.0], [1.0, 0.0], [3.0, 3.0]])
    event = fu.from_curve_indices(2, True, 3, False, P, Q)
    event_copy = event.copy()

    assert event == event_copy
    assert event.dist == event_copy.dist
    assert event is not event_copy


def test_morphing_copy() -> None:
    morphing = u.get_basic_morphing()
    other_morphing = morphing.copy()

    assert other_morphing is not morphing
    assert np.array_equal(other_morphing.P, morphing.P)
    assert np.array_equal(other_morphing.Q, morphing.Q)
    assert other_morphing.dist == morphing.dist

    for event_1, event_2 in zip(morphing.morphing_list, other_morphing.morphing_list):
        assert event_1 == event_2
        assert event_1 is not event_2


def test_morphing_is_monotone() -> None:
    non_monotone_morphing = u.get_basic_morphing()
    assert not non_monotone_morphing.is_monotone()

    monotone_morphing = u.get_basic_morphing_monotone()
    assert monotone_morphing.is_monotone()
