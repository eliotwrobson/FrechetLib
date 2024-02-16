import numpy as np


def leq_with_tolerance(f1: np.floating, f2: np.floating, tol: float = 1e-5) -> bool:
    return f1 + tol >= f2


def generate_curves_random(
    num_pts: int, scaling_factor: float
) -> tuple[np.ndarray, np.ndarray]:
    P = np.random.rand(num_pts, 2) * scaling_factor
    Q = np.random.rand(num_pts, 2) * scaling_factor
    return P, Q


def generate_curves_close(
    num_pts: int, scaling_factor: float
) -> tuple[np.ndarray, np.ndarray]:
    P = np.random.uniform(low=1.0, high=scaling_factor, size=(num_pts, 2))
    Q = P + np.random.uniform(low=0.0, high=0.5, size=(num_pts, 2))
    return P, Q
