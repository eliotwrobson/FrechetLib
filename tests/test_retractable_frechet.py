import numpy as np
from utils import leq_with_tolerance

import frechetlib.retractable_frechet as rf


def test_frechet() -> None:
    n = 100
    P = np.random.rand(n, 2)
    Q = np.random.rand(n, 2)

    frechet_dist, morphing = rf.retractable_ve_frechet(P, Q)

    for event in morphing:
        assert leq_with_tolerance(frechet_dist, event.dist)

    assert morphing[0].i == 0
    assert morphing[0].i_is_vert
    assert morphing[0].j == 0
    assert morphing[0].j_is_vert
    assert morphing[0].t is None
