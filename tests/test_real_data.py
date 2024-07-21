import frechetlib.continuous_frechet as cf
import frechetlib.discrete_frechet as df
import frechetlib.retractable_frechet as rf
import numpy as np
from conftest import get_test_curve

# TODO a lot of these test cases are very similar, try using a fixture to parameterize.


def test_curve_1() -> None:
    P_curve = get_test_curve("01/poly_a.txt")
    Q_curve = get_test_curve("01/poly_b.txt")
    dist_discrete, _ = df.discrete_frechet(P_curve, Q_curve)
    retractable_morphing = rf.retractable_ve_frechet(
        P_curve, Q_curve, None, None, False
    )

    assert np.isclose(dist_discrete, 2.4129857024027306)
    assert np.isclose(retractable_morphing.dist, 1.35)


def test_curve_2() -> None:
    P_curve = get_test_curve("02/poly_a.txt")
    Q_curve = get_test_curve("02/poly_b.txt")
    dist_discrete, _ = df.discrete_frechet(P_curve, Q_curve)
    retractable_morphing = rf.retractable_ve_frechet(
        P_curve, Q_curve, None, None, False
    )

    assert np.isclose(dist_discrete, 1.0440306)
    assert np.isclose(retractable_morphing.dist, 1.0)


def test_curve_3() -> None:
    P_curve = get_test_curve("03/poly_a.txt")
    Q_curve = get_test_curve("03/poly_b.txt")
    dist_discrete, _ = df.discrete_frechet(P_curve, Q_curve)
    retractable_morphing = rf.retractable_ve_frechet(
        P_curve, Q_curve, None, None, False
    )

    assert np.isclose(dist_discrete, 0.8602325267042626)
    assert np.isclose(retractable_morphing.dist, 0.7)


def test_curve_4() -> None:
    P_curve = get_test_curve("04/poly_a.txt")
    Q_curve = get_test_curve("04/poly_b.txt")
    dist_discrete, _ = df.discrete_frechet(P_curve, Q_curve)
    retractable_morphing = rf.retractable_ve_frechet(
        P_curve, Q_curve, None, None, False
    )

    assert np.isclose(dist_discrete, 1.1)
    assert np.isclose(retractable_morphing.dist, 1.1)


def test_curve_5() -> None:
    P_curve = get_test_curve("05/poly_a.txt")
    Q_curve = get_test_curve("05/poly_b.txt")
    dist_discrete, _ = df.discrete_frechet(P_curve, Q_curve)
    retractable_morphing = rf.retractable_ve_frechet(
        P_curve, Q_curve, None, None, False
    )
    ratio, approx_morphing = cf.frechet_c_approx(P_curve, Q_curve, 1.001)

    assert np.isclose(dist_discrete, 1.8442209792777449, atol=0.004)
    assert np.isclose(retractable_morphing.dist, 0.56)

    assert np.isclose(ratio, 1.0)
    assert np.isclose(approx_morphing.dist, 0.712928554361795, atol=0.0004)


def test_curve_6() -> None:
    P_curve = get_test_curve("06/poly_a.txt")
    Q_curve = get_test_curve("06/poly_b.txt")
    dist_discrete, _ = df.discrete_frechet(P_curve, Q_curve)
    retractable_morphing = rf.retractable_ve_frechet(
        P_curve, Q_curve, None, None, False
    )
    ratio, approx_morphing = cf.frechet_c_approx(P_curve, Q_curve, 1.001)

    assert np.isclose(dist_discrete, 1.9224856359432607)
    assert np.isclose(retractable_morphing.dist, 0.9)

    assert np.isclose(ratio, 1.0)
    assert np.isclose(approx_morphing.dist, 0.9212672396766863)


def test_curve_7() -> None:
    P_curve = get_test_curve("07/poly_a.txt")
    Q_curve = get_test_curve("07/poly_b.txt")
    dist_discrete, _ = df.discrete_frechet(P_curve, Q_curve)
    retractable_morphing = rf.retractable_ve_frechet(
        P_curve, Q_curve, None, None, False
    )
    ratio, approx_morphing = cf.frechet_c_approx(P_curve, Q_curve, 1.001)

    assert np.isclose(dist_discrete, 5.30754180388624)
    assert np.isclose(retractable_morphing.dist, 5.30754180388624)

    assert np.isclose(ratio, 1.0)
    assert np.isclose(approx_morphing.dist, 5.30754180388624)


def test_curve_10() -> None:
    P_curve = get_test_curve("10/poly_a.txt")
    Q_curve = get_test_curve("10/poly_b.txt")
    dist_discrete, _ = df.discrete_frechet(P_curve, Q_curve)
    retractable_morphing = rf.retractable_ve_frechet(
        P_curve, Q_curve, None, None, False
    )
    ratio, approx_morphing = cf.frechet_c_approx(P_curve, Q_curve, 1.001)

    assert np.isclose(dist_discrete, 0.9486832980505139)
    assert np.isclose(retractable_morphing.dist, 0.823687767580373)

    # TODO figure out what this number is actually supposed to be
    # assert np.isclose(ratio, 1.0)
    # assert np.isclose(approx_morphing.dist, 0.9486832980505139)


def test_curve_11() -> None:
    P_curve = get_test_curve("11/poly_a.txt")
    Q_curve = get_test_curve("11/poly_b.txt")
    res, morphing = cf.frechet_c_approx(P_curve, Q_curve, 1.01)

    assert np.isclose(morphing.dist, 16.35198236364533)


def test_curve_12() -> None:
    P_curve = get_test_curve("12/poly_a.txt")
    Q_curve = get_test_curve("12/poly_b.txt")
    dist, _ = df.discrete_retractable_frechet(P_curve, Q_curve)

    assert np.isclose(dist, 0.0002966043155438105)
