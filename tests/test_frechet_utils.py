import numpy as np
import pytest
import utils as u

import frechetlib.frechet_utils as fu


def check_morphing_witness(morphing: fu.Morphing) -> None:
    """
    Test helper function to ensure morphing is valid + correctly
    witnessess the given distance.
    """
    target_dist = morphing.dist
    prev_event = None
    saw_witness = False

    morphing_iter = iter(morphing.morphing_list)
    # Skip first iteration because first two events are treated identical
    next(morphing_iter)

    for event in morphing_iter:
        saw_witness = saw_witness or np.isclose(target_dist, event.dist)

        if prev_event is not None:
            # Asserts that consecutive events are contiguous
            assert (event.i - prev_event.i, event.j - prev_event.j) in {
                (0, 1),
                (1, 0),
                (1, 1),
            }

        prev_event = event

    assert saw_witness


@pytest.mark.parametrize(
    "p1,p2,q,distance,t,p_new",
    [
        (
            np.array([-1.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            1.0,
            0.5,
            np.array([0.0, 0.0]),
        ),
        (
            np.array([-1.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([2.0, 0.0]),
            1.0,
            1.0,
            np.array([1.0, 0.0]),
        ),
        (
            np.array([-1.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([-2.0, 0.0]),
            1.0,
            0.0,
            np.array([-1.0, 0.0]),
        ),
    ],
)
def test_line_point_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    q: np.ndarray,
    distance: float,
    t: float,
    p_new: np.ndarray,
) -> None:
    new_dist, new_t, new_p = fu.line_point_distance(p1, p2, q)

    assert 0.0 <= new_t <= 1.0
    assert np.isclose(distance, new_dist)
    assert np.isclose(t, new_t)
    assert np.allclose(p_new, new_p)


def test_convex_comb() -> None:
    # TODO replace this with an actual test of the convex combination function
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


def test_morphing_make_monotone() -> None:
    non_monotone_morphing = u.get_basic_morphing()
    other_morphing = non_monotone_morphing.copy()
    other_morphing.make_monotone()

    assert not non_monotone_morphing.is_monotone()
    assert other_morphing.is_monotone()
    assert len(other_morphing) == len(non_monotone_morphing)
    check_morphing_witness(other_morphing)
    assert other_morphing.dist >= non_monotone_morphing.dist
