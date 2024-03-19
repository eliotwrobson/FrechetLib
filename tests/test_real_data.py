import frechetlib.continuous_frechet as cf
import frechetlib.data as fld
import frechetlib.discrete_frechet as df
import frechetlib.retractable_frechet as rf
import numpy as np
import pytest


@pytest.fixture
def frechet_downloader() -> fld.FrechetDownloader:
    return fld.FrechetDownloader()


def test_curve_1(frechet_downloader: fld.FrechetDownloader) -> None:
    P_curve = frechet_downloader.get_curve("01/poly_a.txt")
    Q_curve = frechet_downloader.get_curve("01/poly_b.txt")
    dist_discrete, _ = df.discrete_frechet(P_curve, Q_curve)
    retractable_morphing = rf.retractable_ve_frechet(P_curve, Q_curve)

    assert np.isclose(dist_discrete, 2.4129857024027306)
    assert np.isclose(retractable_morphing.dist, 1.35)


def test_curve_2(frechet_downloader: fld.FrechetDownloader) -> None:
    P_curve = frechet_downloader.get_curve("02/poly_a.txt")
    Q_curve = frechet_downloader.get_curve("02/poly_b.txt")
    dist_discrete, _ = df.discrete_frechet(P_curve, Q_curve)
    retractable_morphing = rf.retractable_ve_frechet(P_curve, Q_curve)

    assert np.isclose(dist_discrete, 1.0440306)
    assert np.isclose(retractable_morphing.dist, 1.0)


def test_curve_3(frechet_downloader: fld.FrechetDownloader) -> None:
    P_curve = frechet_downloader.get_curve("03/poly_a.txt")
    Q_curve = frechet_downloader.get_curve("03/poly_b.txt")
    dist_discrete, _ = df.discrete_frechet(P_curve, Q_curve)
    retractable_morphing = rf.retractable_ve_frechet(P_curve, Q_curve)

    # TODO the value on Sariel's site is wrong
    # assert np.isclose(dist_discrete, 0.700446286306095)
    assert np.isclose(retractable_morphing.dist, 0.7)


def test_curve_4(frechet_downloader: fld.FrechetDownloader) -> None:
    P_curve = frechet_downloader.get_curve("04/poly_a.txt")
    Q_curve = frechet_downloader.get_curve("04/poly_b.txt")
    dist_discrete, _ = df.discrete_frechet(P_curve, Q_curve)
    retractable_morphing = rf.retractable_ve_frechet(P_curve, Q_curve)

    # TODO the value on Sariel's site is wrong
    # assert np.isclose(dist_discrete, 0.712928554361795)
    # assert np.isclose(retractable_morphing.dist, 0.56)


def test_curve_5(frechet_downloader: fld.FrechetDownloader) -> None:
    P_curve = frechet_downloader.get_curve("05/poly_a.txt")
    Q_curve = frechet_downloader.get_curve("05/poly_b.txt")
    dist_discrete, _ = df.discrete_frechet(P_curve, Q_curve)
    retractable_morphing = rf.retractable_ve_frechet(P_curve, Q_curve)
    ratio, approx_morphing = cf.frechet_c_approx(P_curve, Q_curve, 1.001)

    assert np.isclose(dist_discrete, 1.8442209792777449, atol=0.004)
    assert np.isclose(retractable_morphing.dist, 0.56)

    assert np.isclose(ratio, 1.0)
    assert np.isclose(approx_morphing.dist, 0.712928554361795, atol=0.7124)


def test_curve_6(frechet_downloader: fld.FrechetDownloader) -> None:
    P_curve = frechet_downloader.get_curve("06/poly_a.txt")
    Q_curve = frechet_downloader.get_curve("06/poly_b.txt")
    dist_discrete, _ = df.discrete_frechet(P_curve, Q_curve)
    retractable_morphing = rf.retractable_ve_frechet(P_curve, Q_curve)
    ratio, approx_morphing = cf.frechet_c_approx(P_curve, Q_curve, 1.001)

    assert np.isclose(dist_discrete, 1.9224856359432607)
    assert np.isclose(retractable_morphing.dist, 0.9)

    assert np.isclose(ratio, 1.0)
    assert np.isclose(approx_morphing.dist, 0.9212672396766863)


def test_curve_12(frechet_downloader: fld.FrechetDownloader) -> None:
    P_curve = frechet_downloader.get_curve("12/poly_a.txt")
    Q_curve = frechet_downloader.get_curve("12/poly_b.txt")
    dist, _ = df.discrete_retractable_frechet(P_curve, Q_curve)

    assert np.isclose(dist, 0.0002966043155438105)
