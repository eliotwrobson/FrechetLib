import numba.typed as nbt
import numpy as np

import frechetlib.continuous_frechet as cf
import frechetlib.frechet_utils as fu
import frechetlib.retractable_frechet as rf


def test_get_monotone_morphing_width() -> None:
    # TODO check with Sariel about this test case
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])
    morphing = nbt.List(
        [
            fu.EID(0, True, 0, True, P, Q),
            fu.EID(0, False, 0, True, P, Q),
            fu.EID(0, False, 1, True, P, Q),
            fu.EID(0, False, 2, True, P, Q),
            fu.EID(0, False, 3, True, P, Q),
            fu.EID(0, False, 4, True, P, Q),
            fu.EID(1, True, 4, True, P, Q),
        ]
    )
    _, res = cf.get_monotone_morphing_width(morphing, P, Q)
    assert len(res) == 3


def test_frechet_mono_via_refinement() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])

    P, Q, monotone_morphing, dist, f_exact = cf.frechet_mono_via_refinement(P, Q, 1.01)
    print(len(monotone_morphing))
    # TODO uncomment and finish debugging this
    # assert False


def test_frechet_add_points() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])
    morphing = nbt.List(
        [
            fu.EID(0, True, 0, True, P, Q),
            fu.EID(0, False, 0, True, P, Q),
            fu.EID(0, False, 1, True, P, Q),
            fu.EID(0, False, 2, True, P, Q),
            fu.EID(0, False, 3, True, P, Q),
            fu.EID(0, False, 4, True, P, Q),
            fu.EID(1, True, 4, True, P, Q),
        ]
    )

    P_new, Q_new = cf.add_points_to_make_monotone(P, Q, morphing)
    assert P_new.shape[0] > P.shape[0]


def test_add_points() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])
    _, morphing = rf.retractable_ve_frechet(P, Q)
    new_P, new_Q = cf.add_points_to_make_monotone(P, Q, morphing)
