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
    assert len(res) == len(morphing)


def test_frechet_mono_via_refinement() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])

    new_P, new_Q, monotone_morphing, mono_width, f_exact = (
        cf.frechet_mono_via_refinement(P, Q, 1.01)
    )
    ve_width, ve_morphing = rf.retractable_ve_frechet(P, Q)

    if f_exact:
        assert np.isclose(ve_width, mono_width)
    else:
        assert ve_width <= mono_width

    assert len(monotone_morphing) >= len(ve_morphing)
    assert new_P.shape[0] >= P.shape[0]
    assert new_Q.shape[0] >= Q.shape[0]


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
