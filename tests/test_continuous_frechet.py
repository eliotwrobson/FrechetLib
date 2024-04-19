import frechetlib.continuous_frechet as cf
import frechetlib.data as fld
import frechetlib.retractable_frechet as rf
import numpy as np


def test_frechet_c_compute() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])
    output = cf.frechet_c_compute(P, Q)
    # TODO this test is flaky best on the operating system because of some annoying tiebreak
    # logic (implementation dependant).
    assert np.isclose(output.dist, 0.14142135623730956)


def test_frechet_c_compute_real(frechet_downloader: fld.FrechetDownloader) -> None:
    # Testing with curve number 5
    P_curve = frechet_downloader.get_curve("05/poly_a.txt")
    Q_curve = frechet_downloader.get_curve("05/poly_b.txt")

    _, output_appx = cf.frechet_c_approx(P_curve, Q_curve, 1.01)
    assert np.isclose(output_appx.dist, 0.7134913516143259)

    output_exact = cf.frechet_c_compute(P_curve, Q_curve)
    assert np.isclose(output_exact.dist, 0.712928554361795, atol=0.0005)

    # Testing with curve number 6
    P_curve = frechet_downloader.get_curve("06/poly_a.txt")
    Q_curve = frechet_downloader.get_curve("06/poly_b.txt")

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


def test_add_points_to_make_monotone() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])

    morphing = rf.retractable_ve_frechet(P, Q, None, None, False)
    new_P, new_Q = cf.add_points_to_make_monotone(morphing)

    new_P_expected = np.array(
        [
            [0.0, 0.0],
            [0.15, 0.15],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6],
            [0.7, 0.7],
            [0.85, 0.85],
            [1.0, 1.0],
        ]
    )

    assert np.allclose(new_P, new_P_expected)
    assert np.allclose(new_Q, Q)

    # Extra zeroes because of a quirk with the ve-frechet algo
    new_P_expected = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.15, 0.15],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6],
            [0.7, 0.7],
            [0.85, 0.85],
            [1.0, 1.0],
        ]
    )

    # Assert the flipped side because of code structure
    morphing = rf.retractable_ve_frechet(Q, P, None, None, False)
    new_Q, new_P = cf.add_points_to_make_monotone(morphing)

    assert np.allclose(new_P, new_P_expected)
    assert np.allclose(new_Q, Q)


def test_add_points() -> None:
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])

    new_P_expected = np.array(
        [
            [0.0, 0.0],
            [0.15, 0.15],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6],
            [0.7, 0.7],
            [0.85, 0.85],
            [1.0, 1.0],
        ]
    )

    morphing = rf.retractable_ve_frechet(P, Q, None, None, False)
    new_P, new_Q = cf.add_points_to_make_monotone(morphing)

    assert np.allclose(new_P, new_P_expected)
    assert np.allclose(new_Q, Q)

    # Check the same but flipping the arguments
    morphing = rf.retractable_ve_frechet(Q, P, None, None, False)
    new_Q, new_P = cf.add_points_to_make_monotone(morphing)

    # We don't check for new_P because of an edge case that produces
    # a slightly different morphing. This isn't a bug
    assert new_P.shape[0] > P.shape[0]
    assert np.allclose(new_Q, Q)
