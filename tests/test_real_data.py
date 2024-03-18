import frechetlib.data as fld
import frechetlib.discrete_frechet as df
import numpy as np
import pytest


@pytest.fixture
def frechet_downloader() -> fld.FrechetDownloader:
    return fld.FrechetDownloader()


# TODO this fails because Sariel made the same uploading mistake everywhere on
# his site. Let him know so he can fix it.
@pytest.mark.xfail
def test_curve_1(frechet_downloader: fld.FrechetDownloader) -> None:
    P_curve = frechet_downloader.get_curve("01/poly_a.txt")
    Q_curve = frechet_downloader.get_curve("01/poly_b.txt")
    dist_discrete, _ = df.discrete_frechet(P_curve, Q_curve)
    dist_retract, _ = df.discrete_retractable_frechet(P_curve, Q_curve)

    assert np.isclose(dist_discrete, 1.35)
    assert np.isclose(dist_retract, 1.35)


def test_curve_12(frechet_downloader: fld.FrechetDownloader) -> None:
    P_curve = frechet_downloader.get_curve("12/poly_a.txt")
    Q_curve = frechet_downloader.get_curve("12/poly_b.txt")
    dist, _ = df.discrete_retractable_frechet(P_curve, Q_curve)

    assert np.isclose(dist, 0.0002966043155438105)
