import frechetlib.geometry_utils as gu
import numpy as np
import pytest


@pytest.mark.parametrize(
    "p1,p2,q,distance,t,p_new",
    [
        (
            np.array([-1.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            1.0,
            0.5,
            [0.0, 0.0],
        ),
        (
            np.array([-1.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([2.0, 0.0]),
            1.0,
            1.0,
            [1.0, 0.0],
        ),
        (
            np.array([-1.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([-2.0, 0.0]),
            1.0,
            0.0,
            [-1.0, 0.0],
        ),
    ],
)
def test_line_point_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    q: np.ndarray,
    distance: float,
    t: float,
    p_new: list[float],
) -> None:
    other_res = gu.LinePointDistance(p1, p2, q)

    assert 0.0 <= other_res.get_t() <= 1.0
    assert np.isclose(distance, other_res.get_distance())
    assert np.isclose(t, other_res.get_t())
    assert p_new == list(other_res.get_p().get_coords())


def test_convex_comb() -> None:
    # TODO replace this with an actual test of the convex combination function
    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0, 1.0])
    q = np.array([1.0, 1.0])
    res = gu.LinePointDistance(p1, p2, q)
    assert 0.0 <= res.get_t() <= 1.0
