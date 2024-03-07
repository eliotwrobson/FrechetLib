import frechetlib.frechet_utils as fu
import frechetlib.retractable_frechet as rf
import numpy as np
import pytest
import utils as u


def example_3():
    P = np.array(
        [
            [0.0, 0],
            [1.1, 0.0],
            [1.1, 0.1],
            [1.0, 0.1],
            [1.0, 0.2],
            [1.2, 0.2],
            [1.2, 0.0],
            [2.0, 0.0],
        ]
    )

    Q = np.array(
        [
            [0.0, 0.3],
            [0.4, 0.3],
            [0.4, 0.6],
            [0.3, 0.6],
            [0.3, 0.7],
            [0.5, 0.7],
            [0.5, 0.3],
            [2.0, 0.3],
        ]
    )

    R = np.array(
        [
            [2.0, 2.3],
            [2.4, 2.3],
            [2.4, 2.6],
            [2.3, 2.6],
            [2.3, 2.7],
            [2.5, 2.7],
            [2.5, 2.3],
            [4.0, 2.3],
        ]
    )

    return P, Q, R


def test_frechet_dist_upper_bound() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])
    res = fu.frechet_dist_upper_bound(P, Q)

    assert np.isclose(fu.frechet_width_approx(P), 0.0)
    assert np.isclose(fu.frechet_width_approx(Q), 0.28284271247461906)
    assert np.isclose(res, 0.28284271247461906)


def test_morphing_combine() -> None:
    scaling_factor = 100.0
    num_pts = 5
    d = 2
    np.random.seed(12345)
    # P = np.random.uniform(low=1.0, high=scaling_factor, size=(num_pts, d))
    # Q = np.random.uniform(low=1.0, high=scaling_factor, size=(num_pts, d))
    # R = np.random.uniform(low=1.0, high=scaling_factor, size=(num_pts, d))

    P, Q, R = example_3()

    morphing_1 = rf.retractable_ve_frechet(P, Q)
    morphing_2 = rf.retractable_ve_frechet(Q, R)

    # Apparently these need to be monotone for this to work
    morphing_1.make_monotone()
    morphing_2.make_monotone()

    res = fu.morphing_combine(morphing_2, morphing_1)

    # Magic numbers I got by running the same data through
    # Sariel's code
    assert np.isclose(morphing_1.dist, 0.7071067811865475)
    assert np.isclose(morphing_2.dist, 2.82842712474619)

    assert np.isclose(res.dist, 3.047950130825634)
    assert np.allclose(res.P, P)
    assert np.allclose(res.Q, R)


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


def test_morphing_flip() -> None:
    P, Q = u.generate_curves_close(100, 100.0)
    morphing = rf.retractable_ve_frechet(P, Q)
    orig_morphing = morphing.copy()
    morphing.flip()

    assert np.allclose(P, morphing.Q)
    assert np.allclose(Q, morphing.P)

    for orig_event, new_event in zip(
        orig_morphing.morphing_list, morphing.morphing_list
    ):
        assert orig_event.i == new_event.j
        assert orig_event.j == new_event.i


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
    event = fu.from_curve_indices(2, True, 2, False, P, Q)
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
