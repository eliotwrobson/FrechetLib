import frechetlib.frechet_utils as fu
import numba.typed as nbt
import numpy as np


def leq_with_tolerance(f1: np.floating, f2: np.floating, tol: float = 1e-5) -> bool:
    return f1 + tol >= f2


# https://github.com/TvoroG/pytest-lazy-fixture/issues/65
def generate_curves_random(
    num_pts: int, scaling_factor: float
) -> tuple[np.ndarray, np.ndarray]:
    P = np.random.rand(num_pts, 2) * scaling_factor
    Q = np.random.rand(num_pts, 2) * scaling_factor
    return P, Q


def generate_curves_close(
    num_pts: int, scaling_factor: float, d: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    P = np.random.uniform(low=1.0, high=scaling_factor, size=(num_pts, d))
    Q = P + np.random.uniform(low=0.0, high=scaling_factor / 100.0, size=(num_pts, d))
    return P, Q


def get_basic_morphing() -> fu.Morphing:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])
    morphing_list = nbt.List(
        [
            fu.from_curve_indices(0, True, 0, True, P, Q, None, None),
            fu.from_curve_indices(0, False, 0, True, P, Q, None, None),
            fu.from_curve_indices(0, False, 1, True, P, Q, None, None),
            fu.from_curve_indices(0, False, 2, True, P, Q, None, None),
            fu.from_curve_indices(0, False, 3, True, P, Q, None, None),
            fu.from_curve_indices(0, False, 4, True, P, Q, None, None),
            fu.from_curve_indices(1, True, 4, True, P, Q, None, None),
        ]
    )

    dist = fu.get_frechet_dist_from_morphing_list(morphing_list)
    return fu.Morphing(morphing_list, P, Q, dist)


def get_basic_morphing_monotone() -> fu.Morphing:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    morphing_list = nbt.List(
        [
            fu.from_curve_indices(0, True, 0, True, P, Q, None, None),
            fu.from_curve_indices(0, False, 0, True, P, Q, None, None),
            fu.from_curve_indices(0, False, 1, True, P, Q, None, None),
            fu.from_curve_indices(0, False, 2, True, P, Q, None, None),
            fu.from_curve_indices(1, True, 2, True, P, Q, None, None),
        ]
    )

    dist = fu.get_frechet_dist_from_morphing_list(morphing_list)
    return fu.Morphing(morphing_list, P, Q, dist)
