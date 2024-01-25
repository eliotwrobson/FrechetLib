import numpy as np


def leq_with_tolerance(f1: np.floating, f2: np.floating, tol: float = 1e-5) -> bool:
    return f1 + tol >= f2
