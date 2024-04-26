import numpy as np
from numba import njit  # type: ignore[attr-defined]

import frechetlib.frechet_utils as fu
import frechetlib.retractable_frechet as rf


# NOTE Needs to be a separate algorithm from the other
# continuous Frechet ones because the additive property
# could cause non-convergence.
@njit
def sweep_frechet_compute_refine_mono(P: np.ndarray, Q: np.ndarray) -> fu.Morphing:
    ell = P.shape[0] + Q.shape[0]

    rate_limit = 0.01

    while True:
        morphing = rf.retractable_ve_frechet(P, Q, None, None, True)

        if morphing.is_monotone():
            break

        error = morphing.copy().make_monotone()
        rate = error / ell

        if rate < rate_limit:
            break

        P, Q = fu.add_points_to_make_monotone(morphing)

    return morphing
