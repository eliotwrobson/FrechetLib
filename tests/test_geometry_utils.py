import numpy as np
import pytest

import frechetlib.geometry_utils as gu


@pytest.mark.parametrize(
    "p1,p2,q,distance,t,p_new",
    [
        (
            gu.Point([-1.0, 0.0]),
            gu.Point([1.0, 0.0]),
            gu.Point([0.0, 1.0]),
            1.0,
            0.5,
            gu.Point([0.0, 0.0]),
        ),
        (
            gu.Point([-1.0, 0.0]),
            gu.Point([1.0, 0.0]),
            gu.Point([2.0, 0.0]),
            1.0,
            1.0,
            gu.Point([1.0, 0.0]),
        ),
        (
            gu.Point([-1.0, 0.0]),
            gu.Point([1.0, 0.0]),
            gu.Point([-2.0, 0.0]),
            1.0,
            0.0,
            gu.Point([-1.0, 0.0]),
        ),
    ],
)
def test_line_point_distance(
    p1: gu.Point,
    p2: gu.Point,
    q: gu.Point,
    distance: float,
    t: float,
    p_new: gu.Point,
) -> None:
    other_res = gu.LinePointDistance(p1, p2, q)

    assert 0.0 <= other_res.get_t() <= 1.0
    assert np.isclose(distance, other_res.get_distance())
    assert np.isclose(t, other_res.get_t())
    assert p_new.is_close(other_res.get_p())


def test_convex_comb() -> None:
    # TODO replace this with an actual test of the convex combination function
    p1 = gu.Point([0.0, 0.0])
    p2 = gu.Point([1.0, 1.0])
    q = gu.Point([1.0, 1.0])
    res = gu.LinePointDistance(p1, p2, q)
    assert 0.0 <= res.get_t() <= 1.0


def test_eid_copy() -> None:
    # Contents here are not really important for this test
    P = gu.numpy_to_point_list(np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]]))
    Q = gu.numpy_to_point_list(np.array([[0.0, 1.0], [1.0, 0.0], [3.0, 3.0]]))
    event = gu.from_curve_indices(2, True, 2, False, P, Q, None, None).get_event()
    event_copy = event.copy()

    assert event == event_copy
    assert event.get_dist() == event_copy.get_dist()
    assert event is not event_copy
