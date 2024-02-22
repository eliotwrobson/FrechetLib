import numba.typed as nbt
import numpy as np
import utils as u

import frechetlib.continuous_frechet as cf
import frechetlib.frechet_utils as fu
import frechetlib.retractable_frechet as rf


def test_add_points() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])
    morphing = rf.retractable_ve_frechet(P, Q)
    print(len(morphing))
    new_P, new_Q = cf.add_points_to_make_monotone(morphing)
    new_morphing = rf.retractable_ve_frechet(P, Q)

    for event in new_morphing:
        print(event.i, event.i_is_vert, event.j, event.j_is_vert)

    print(new_P)
    print(new_Q)
    print(new_morphing.dist)
    assert False


def test_frechet_mono_via_refinement() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])

    monotone_morphing, f_exact = cf.frechet_mono_via_refinement(P, Q, 1.01)
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


def test_frechet_add_points() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])
    morphing = nbt.List(
        [
            fu.from_curve_indices(0, True, 0, True, P, Q),
            fu.from_curve_indices(0, False, 0, True, P, Q),
            fu.from_curve_indices(0, False, 1, True, P, Q),
            fu.from_curve_indices(0, False, 2, True, P, Q),
            fu.from_curve_indices(0, False, 3, True, P, Q),
            fu.from_curve_indices(0, False, 4, True, P, Q),
            fu.from_curve_indices(1, True, 4, True, P, Q),
        ]
    )

    P_new, Q_new = cf.add_points_to_make_monotone(P, Q, morphing)
    assert P_new.shape[0] > P.shape[0]
