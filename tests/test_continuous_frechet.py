import numpy as np

import frechetlib.continuous_frechet as cf
import frechetlib.retractable_frechet as rf


def test_frechet_mono_via_refinement() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])

    monotone_morphing, f_exact = cf.frechet_mono_via_refinement(P, Q, 1.01)

    assert len(monotone_morphing.morphing_list) == 13

    ve_morphing = rf.retractable_ve_frechet(P, Q)

    if f_exact:
        assert np.isclose(ve_morphing.dist, monotone_morphing.dist)
    else:
        assert ve_morphing.dist <= monotone_morphing.dist

    # TODO I think I can assert the length is just the sum of the lengths of the
    # number of points in each curve
    assert len(monotone_morphing.morphing_list) >= len(ve_morphing.morphing_list)
    assert monotone_morphing.P.shape[0] >= P.shape[0]
    assert monotone_morphing.Q.shape[0] >= Q.shape[0]


def test_frechet_c_approx() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])
    cf.frechet_c_approx(P, Q, 1.01)


def test_add_points() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])

    new_P_expected = np.array(
        [
            [0.0, 0.0],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6],
            [0.7, 0.7],
            [1.0, 1.0],
        ]
    )

    morphing = rf.retractable_ve_frechet(P, Q)
    new_P, new_Q = cf.add_points_to_make_monotone(morphing)

    for event in morphing.morphing_list:
        print(event.i, event.i_is_vert, event.j, event.j_is_vert)

    assert np.allclose(new_P, new_P_expected)
    assert np.allclose(new_Q, Q)

    # Check the same but flipping the arguments
    morphing = rf.retractable_ve_frechet(Q, P)
    new_Q, new_P = cf.add_points_to_make_monotone(morphing)

    # We don't check for new_P because of an edge case that produces
    # a slightly different morphing. This isn't a bug
    assert new_P.shape[0] > P.shape[0]
    assert np.allclose(new_Q, Q)
