import numpy as np

import frechetlib.retractable_frechet as rf


def test_classes() -> None:
    rf.EventPoint(
        np.ndarray(2),
        0,
        rf.EventType.POINT_VERTEX,
        0.0,
        np.ndarray(2),
    )
