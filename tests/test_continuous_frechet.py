import numpy as np
import pytest
from conftest import get_test_curve

import frechetlib.continuous_frechet as cf
import frechetlib.retractable_frechet as rf

# def test_frechet_c_compute() -> None:
#     P = np.array([[0.0, 0.0], [1.0, 1.0]])
#     Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])
#     output = cf.frechet_c_compute(P, Q)
#     # TODO this test is flaky best on the operating system because of some annoying tiebreak
#     # logic (implementation dependant).
#     assert np.isclose(output.get_dist(), 0.14142135623730956)


@pytest.mark.parametrize(
    ("curve_num", "expected_dist_approx", "expected_dist_exact", "atol"),
    [
        ("18", 2.828116009218858, 2.828116009218858, 0.0002),
        ("17", 2.9832867780352594, 2.9832867780352594, 0.0002),
        ("16", 1.1585808408611906, 1.1585808408611906, 0.0002),
        ("15", 0.6545806432759624, 0.6545806432759624, 0.0002),
        ("14", None, 0.921199917014942, 0.0002),
        ("13", 4.8, 4.8, 0.0002),
        ("12", None, 0.0002966043155438105, 0.0000002),
        ("11", None, 1.2481323161556228, 0.0002),
        ("10", None, 0.823687767580373, 0.0002),
        ("09", None, 0.43467993585364817, 0.0002),
        ("08", None, 0.39, 0.0002),
        # Number switch
        ("01", 1.35, 1.35, 0.0005),
        ("02", 1.0, 1.0, 0.0005),
        ("03", 0.7004463076591492, 0.7, 0.0005),
        ("04", None, 1.1, 0.0005),
        ("05", 0.7134913516143259, 0.712928554361795, 0.0005),
        ("06", 0.9228858795210783, 0.9212672396766863, 0.0002),
        ("07", 5.30754180388624, 5.30754180388624, 0.0002),
    ],
)
def test_frechet_c_compute_real(
    curve_num: str,
    expected_dist_approx: float | None,
    expected_dist_exact: float,
    atol: float,
) -> None:
    P_curve = get_test_curve(f"{curve_num}/poly_a.txt")
    Q_curve = get_test_curve(f"{curve_num}/poly_b.txt")

    if expected_dist_approx is not None:
        _, output_appx = cf.frechet_c_approx(P_curve, Q_curve, 1.01)
        assert np.isclose(output_appx.get_dist(), expected_dist_approx)

    output_exact = cf.frechet_c_compute(P_curve, Q_curve)
    assert np.isclose(output_exact.get_dist(), expected_dist_exact, atol=atol)


def test_frechet_c_approx() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])
    ratio, output_morphing = cf.frechet_c_approx(P, Q, 1.01)

    assert np.isclose(output_morphing.get_dist(), 0.14142135623730956, atol=0.0003)
    assert np.isclose(ratio, 1.0)


def test_frechet_mono_via_refinement() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])

    monotone_morphing, f_exact = cf.frechet_mono_via_refinement(P, Q, 1.01)

    ve_morphing = rf.retractable_ve_frechet(P, Q, None, None, False)

    if f_exact:
        assert np.isclose(ve_morphing.get_dist(), monotone_morphing.get_dist())
    else:
        assert ve_morphing.get_dist() <= monotone_morphing.get_dist()

    # TODO I think I can assert the length is just the sum of the lengths of the
    # number of points in each curve
    assert len(monotone_morphing.get_morphing_list()) >= len(
        ve_morphing.get_morphing_list()
    )
    assert monotone_morphing.get_P().shape[0] >= P.shape[0]
    assert monotone_morphing.get_Q().shape[0] >= Q.shape[0]
