import frechetlib.frechet_utils as fu
import frechetlib.retractable_frechet as rf
import numba.typed as nbt
import numpy as np
import pytest


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


def test_morphing_combine_manual() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])

    # Morphing P with itself
    P_self_morphing_list = nbt.List(
        [
            fu.EID(0, True, 0, True, P[0], P[0], 0.0, 0.0, 0.0),
            fu.EID(1, True, 1, True, P[1], P[1], 0.0, 0.0, 0.0),
        ]
    )

    P_self_morphing = fu.Morphing(P_self_morphing_list, P, P, 0.0)

    # Morphing Q with itself
    Q_self_morphing_list = nbt.List(
        [
            fu.EID(0, True, 0, True, Q[0], Q[0], 0.0, 0.0, 0.0),
            fu.EID(1, True, 1, True, Q[1], Q[1], 0.0, 0.0, 0.0),
            fu.EID(2, True, 2, True, Q[2], Q[2], 0.0, 0.0, 0.0),
            fu.EID(3, True, 3, True, Q[3], Q[3], 0.0, 0.0, 0.0),
            fu.EID(4, True, 4, True, Q[4], Q[4], 0.0, 0.0, 0.0),
        ]
    )

    Q_self_morphing = fu.Morphing(Q_self_morphing_list, Q, Q, 0.0)

    P_refined = np.array(
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

    dist = 0.14142135623730956
    # Middle Morphing. Using P points in multiple places to avoid having to redefine points
    middle_morphing_list = nbt.List(
        [
            # 1
            fu.EID(0, True, 0, True, P_refined[0], P_refined[0], 0.0, 0.0, 0.0),
            # 2
            fu.EID(0, False, 0, True, P_refined[0], P_refined[0], 0.0, 0.0, 0.0),
            # 3
            fu.EID(1, True, 0, False, P_refined[1], P_refined[1], 0.0, 0.3, 0.0),
            # 4
            fu.EID(2, True, 0, False, P_refined[2], P_refined[2], 0.0, 0.6, 0.0),
            # 5, only one witnessing a difference
            fu.EID(2, False, 1, True, P_refined[3], P_refined[4], 1.0, 0.0, dist),
            # 6, flips backwards
            fu.EID(2, False, 2, True, P_refined[3], P_refined[2], 1.0, 0.0, dist),
            # 7
            fu.EID(3, True, 2, False, P_refined[3], P_refined[3], 0.0, 0.25, 0.0),
            # 8
            fu.EID(4, True, 2, False, P_refined[4], P_refined[4], 0.0, 0.5, 0.0),
            # 9
            fu.EID(5, True, 2, False, P_refined[5], P_refined[5], 0.0, 0.75, 0.0),
            # 10
            fu.EID(6, True, 2, False, P_refined[6], P_refined[6], 0.0, 1.0, 0.0),
            # 11
            fu.EID(6, False, 3, True, P_refined[6], P_refined[6], 0.0, 0.0, 0.0),
            # 12
            fu.EID(7, True, 3, False, P_refined[7], P_refined[7], 0.0, 0.5, 0.0),
            # 13
            fu.EID(8, True, 4, True, P_refined[8], P_refined[8], 0.0, 0.0, 0.0),
        ]
    )

    middle_morphing = fu.Morphing(middle_morphing_list, P_refined, Q, dist)

    P_self_morphing_prm = np.array(
        [
            [0.0, 1.4142135623730951],
            [0.0, 1.4142135623730951],
        ]
    )

    assert np.allclose(P_self_morphing.get_prm(), P_self_morphing_prm)

    # Pulled from Sariel's code
    middle_morphing_prm = np.array(
        [
            [
                0.0,
                0.0,
                0.21213203435596426,
                0.4242640687119285,
                0.565685424949238,
                0.565685424949238,
                0.565685424949238,
                0.7071067811865475,
                0.8485281374238569,
                0.9899494936611664,
                0.9899494936611664,
                1.2020815280171306,
                1.414213562373095,
            ],
            [
                0.0,
                0.0,
                0.21213203435596426,
                0.4242640687119285,
                0.7071067811865476,
                0.9899494936611666,
                1.131370849898476,
                1.2727922061357857,
                1.414213562373095,
                1.5556349186104046,
                1.5556349186104046,
                1.7677669529663689,
                1.9798989873223332,
            ],
        ]
    )

    assert np.allclose(middle_morphing_prm, middle_morphing.get_prm())

    first_combined = fu.morphing_combine(middle_morphing, P_self_morphing)

    assert np.isclose(first_combined.dist, 0.14142135623730956)

    first_combined.make_monotone()

    assert np.isclose(first_combined.dist, 0.14142135623730956)

    assert False


def test_morphing_combine_again() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    p_indices = [0, 1]

    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])
    q_indices = [0, 1, 2, 3, 4]

    morphing_p = cf.frechet_c_mono_approx_subcurve(P, P, p_indices)
    morphing_q = cf.frechet_c_mono_approx_subcurve(Q, Q, q_indices)
    morphing, _ = cf.frechet_mono_via_refinement(P, Q, 1.0025)

    assert np.isclose(morphing_p.dist, 0.0)
    assert np.isclose(morphing_q.dist, 0.0)
    assert np.isclose(morphing.dist, 0.14169756982371037)

    morphing_p.make_monotone()
    morphing_q.make_monotone()

    # Should be the same after making monotone
    assert np.isclose(morphing_p.dist, 0.0)
    assert np.isclose(morphing_q.dist, 0.0)

    first_morphing = fu.morphing_combine(morphing, morphing_p)
    # TODO this might be close enough that it doesn't matter?
    assert np.isclose(first_morphing.dist, 0.14142135623730956)

    first_morphing.make_monotone()
    # assert np.isclose(first_morphing.dist, 0.14142135623730956)

    first_morphing.flip()
    # prm = first_morphing.get_prm()
    # print(prm)
    print("Done with first monotone")
    output_morphing = fu.morphing_combine(first_morphing, morphing_q)
    assert np.isclose(output_morphing.dist, 0.14142135623730956)


def test_morphing_make_monotone_nontrivial() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])

    ve_morphing = rf.retractable_ve_frechet(P, Q)
    assert np.isclose(ve_morphing.dist, 0.0)
    monotone_morphing = ve_morphing.copy()
    monotone_morphing.make_monotone()
    assert np.isclose(monotone_morphing.dist, 0.28284271247461906)

    new_P, new_Q = cf.add_points_to_make_monotone(ve_morphing)

    # Compute new ve frechet distance for curves
    ve_morphing = rf.retractable_ve_frechet(new_P, new_Q)

    assert np.isclose(ve_morphing.dist, 0.14142135623730948)

    print("Starting the actual run")
    # Make monotone
    monotone_morphing_2 = ve_morphing.copy()
    monotone_morphing_2.make_monotone()

    # TODO this fails because, on th 6th point, the first curve has
    # to double back on itself, which causes the algo to take forever
    # to converge.
    # assert np.isclose(monotone_morphing_2.dist, 0.14142135623730948)


def test_get_prm() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])

    morphing, _ = cf.frechet_mono_via_refinement(P, Q, 1.0025)

    prm = morphing.get_prm()

    fu.print_prm(prm)
    assert False


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
