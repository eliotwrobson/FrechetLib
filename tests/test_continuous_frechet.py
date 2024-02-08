import numba.typed as nbt
import numpy as np

import frechetlib.continuous_frechet as cf
import frechetlib.retractable_frechet as rf


def test_get_monotone_morphing_width() -> None:
    # TODO check with Sariel about this test case
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])
    morphing = nbt.List(
        [
            rf.EID(0, True, 0, True, P, Q),
            rf.EID(0, False, 0, True, P, Q),
            rf.EID(0, False, 1, True, P, Q),
            rf.EID(0, False, 2, True, P, Q),
            rf.EID(0, False, 3, True, P, Q),
            rf.EID(0, False, 4, True, P, Q),
            rf.EID(1, True, 4, True, P, Q),
        ]
    )
    _, res = cf.get_monotone_morphing_width(morphing)
    assert len(res) == 3


def test_frechet_add_points() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])
    morphing = nbt.List(
        [
            rf.EID(0, True, 0, True, P, Q),
            rf.EID(0, False, 0, True, P, Q),
            rf.EID(0, False, 1, True, P, Q),
            rf.EID(0, False, 2, True, P, Q),
            rf.EID(0, False, 3, True, P, Q),
            rf.EID(0, False, 4, True, P, Q),
            rf.EID(1, True, 4, True, P, Q),
        ]
    )

    ((P_new_points, P_indices), (Q_new_points, Q_indices)) = (
        cf.add_points_to_make_monotone(P, Q, morphing)
    )

    np.insert(P, P_new_points, P_indices)
    np.insert(Q, Q_new_points, Q_indices)
    print(P_indices)
    print(Q_indices)
    assert False
