import numba as nb
import numpy as np


@nb.njit
def line_point_distance(
    p1: np.ndarray, p2: np.ndarray, q: np.ndarray
) -> tuple[float, float]:
    """
    Based on: https://stackoverflow.com/a/1501725/2923069
    """
    # Return minimum distance between line segment p1-p2 and point q
    l2 = np.linalg.norm(p1 - p2)  # i.e. |p2-p1|^2 -  avoid a sqrt
    if np.isclose(l2, 0.0):  # p1 == p2 case
        return float(np.linalg.norm(q - p1)), 0.0
    # Consider the line extending the segment, parameterized as v + t (p2 - p1).
    # We find projection of point q onto the line.
    # It falls where t = [(q-p1) . (p2-p1)] / |p2-p1|^2
    # We clamp t from [0,1] to handle points outside the segment vw.
    t = np.dot(q - p1, p2 - p1) / l2

    if t <= 0.0:
        return float(np.linalg.norm(p1 - q)), t
    elif t >= 1.0:
        return float(np.linalg.norm(p2 - q)), t

    return float(np.linalg.norm(q - (p1 + t * (p2 - p1)))), t
