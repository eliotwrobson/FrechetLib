import numpy as np
import pytest
from conftest import get_test_curve

import frechetlib.continuous_frechet as cf
import frechetlib.retractable_frechet as rf


def test_frechet_mono_via_refinement() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])

    monotone_morphing = cf.frechet_mono_via_refinement(P, Q, 1.01)

    ve_morphing = rf.retractable_ve_frechet(P, Q, None, None, False)

    # if f_exact:
    # assert np.isclose(ve_morphing.get_dist(), monotone_morphing.get_dist())
    # else:
    assert ve_morphing.get_dist() <= monotone_morphing.get_dist()
    assert np.isclose(ve_morphing.get_dist(), 0.0)

    # TODO I think I can assert the length is just the sum of the lengths of the
    # number of points in each curve
    assert len(monotone_morphing.get_morphing_list()) >= len(
        ve_morphing.get_morphing_list()
    )
    assert len(monotone_morphing.get_P()) >= P.shape[0]
    assert len(monotone_morphing.get_Q()) >= Q.shape[0]


# def test_frechet_c_compute() -> None:
#     P = np.array([[0.0, 0.0], [1.0, 1.0]])
#     Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])
#     output = cf.frechet_c_compute(P, Q)
#     # TODO this test is flaky best on the operating system because of some annoying tiebreak
#     # logic (implementation dependant).
#     assert np.isclose(output.get_dist(), 0.14142135623730956)


@pytest.mark.parametrize(
    ("curve_num", "expected_dist_approx"),
    [
        ("20", 8.37797726526284),
        ("19", 6.839958603290227),
        ("18", 2.828116009218858),
        ("17", 2.9832867780352594),
        ("16", 1.1585808408611906),
        ("15", 0.6545806432759624),
        ("13", 4.8),
        # Number switch
        ("07", 5.30754180388624),
        ("06", 0.9228858795210783),
        ("05", 0.7134913516143259),
        ("03", 0.7004463076591492),
        ("02", 1.0),
        ("01", 1.35),
    ],
)
def test_frechet_c_approx_real(
    curve_num: str,
    expected_dist_approx: float,
) -> None:
    P_curve = get_test_curve(f"{curve_num}/poly_a.txt")
    Q_curve = get_test_curve(f"{curve_num}/poly_b.txt")

    _, output_appx = cf.frechet_c_approx(P_curve, Q_curve, 1.01)
    assert np.isclose(output_appx.get_dist(), expected_dist_approx)


@pytest.mark.parametrize(
    ("curve_num", "expected_dist_exact", "atol"),
    [
        ("19", 6.839958603290227, 0.0002),
        ("18", 2.828116009218858, 0.0002),
        ("17", 2.9832867780352594, 0.0002),
        ("16", 1.1585808408611906, 0.0002),
        ("15", 0.6545806432759624, 0.0002),
        ("14", 0.921199917014942, 0.0002),
        ("13", 4.8, 0.0002),
        ("12", 0.0002966043155438105, 0.0000002),
        # ("11", 1.2481323161556228, 0.0002),
        ("10", 0.823687767580373, 0.0002),
        # ("09", 0.43467993585364817, 0.0002),
        # Number switch
        ("07", 5.30754180388624, 0.0002),
        ("06", 0.9212672396766863, 0.0002),
        ("05", 0.712928554361795, 0.0005),
        ("04", 1.1, 0.0005),
        ("03", 0.7, 0.0005),
        ("02", 1.0, 0.0005),
        ("01", 1.35, 0.0005),
    ],
)
def test_frechet_c_compute_real(
    curve_num: str,
    expected_dist_exact: float,
    atol: float,
) -> None:
    P_curve = get_test_curve(f"{curve_num}/poly_a.txt")
    Q_curve = get_test_curve(f"{curve_num}/poly_b.txt")

    output_exact = cf.frechet_c_compute(P_curve, Q_curve)
    assert np.isclose(output_exact.get_dist(), expected_dist_exact, atol=atol)


def test_frechet_c_approx() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])
    ratio, output_morphing = cf.frechet_c_approx(P, Q, 1.01)

    assert np.isclose(output_morphing.get_dist(), 0.14142135623730956, atol=0.0003)
    assert np.isclose(ratio, 1.0)
