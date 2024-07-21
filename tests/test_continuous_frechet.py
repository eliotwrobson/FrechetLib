import frechetlib.continuous_frechet as cf
import frechetlib.retractable_frechet as rf
import numpy as np
from conftest import get_test_curve


def test_frechet_c_compute() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])
    output = cf.frechet_c_compute(P, Q)
    # TODO this test is flaky best on the operating system because of some annoying tiebreak
    # logic (implementation dependant).
    assert np.isclose(output.dist, 0.14142135623730956)


def test_frechet_c_compute_real() -> None:
    # Testing with curve number 5
    P_curve = get_test_curve("05/poly_a.txt")
    Q_curve = get_test_curve("05/poly_b.txt")

    _, output_appx = cf.frechet_c_approx(P_curve, Q_curve, 1.01)
    assert np.isclose(output_appx.dist, 0.7134913516143259)

    output_exact = cf.frechet_c_compute(P_curve, Q_curve)
    assert np.isclose(output_exact.dist, 0.712928554361795, atol=0.0005)

    # Testing with curve number 6
    P_curve = get_test_curve("06/poly_a.txt")
    Q_curve = get_test_curve("06/poly_b.txt")

    _, output_appx = cf.frechet_c_approx(P_curve, Q_curve, 1.01)
    assert np.isclose(output_appx.dist, 0.9228858795210783)

    output_exact = cf.frechet_c_compute(P_curve, Q_curve)
    assert np.isclose(output_exact.dist, 0.9212672396766863, atol=0.0002)


def test_frechet_c_approx() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])
    ratio, output_morphing = cf.frechet_c_approx(P, Q, 1.01)

    assert np.isclose(output_morphing.dist, 0.14142135623730956, atol=0.0003)
    assert np.isclose(ratio, 1.0)


def test_frechet_mono_via_refinement() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])

    monotone_morphing, f_exact = cf.frechet_mono_via_refinement(P, Q, 1.01)

    ve_morphing = rf.retractable_ve_frechet(P, Q, None, None, False)

    if f_exact:
        assert np.isclose(ve_morphing.dist, monotone_morphing.dist)
    else:
        assert ve_morphing.dist <= monotone_morphing.dist

    # TODO I think I can assert the length is just the sum of the lengths of the
    # number of points in each curve
    assert len(monotone_morphing.morphing_list) >= len(ve_morphing.morphing_list)
    assert monotone_morphing.P.shape[0] >= P.shape[0]
    assert monotone_morphing.Q.shape[0] >= Q.shape[0]
