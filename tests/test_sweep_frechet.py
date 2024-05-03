import frechetlib.data as fld
import frechetlib.sweep_frechet as sf


def test_sweep_frechet_refinement(frechet_downloader: fld.FrechetDownloader) -> None:
    P_curve = frechet_downloader.get_curve("03/poly_a.txt")
    Q_curve = frechet_downloader.get_curve("03/poly_b.txt")
    sweep_morphing = sf.sweep_frechet_compute_refine_mono(P_curve, Q_curve)

    print(len(sweep_morphing.morphing_list))
    dist = sf.sweep_frechet_compute_lower_bound(P_curve, Q_curve)
    print(dist)
    # assert False
