import frechetlib.sweep_frechet as sf
from conftest import get_test_curve


def test_sweep_frechet_refinement() -> None:
    P_curve = get_test_curve("03/poly_a.txt")
    Q_curve = get_test_curve("03/poly_b.txt")
    sweep_morphing = sf.sweep_frechet_compute_refine_mono(P_curve, Q_curve)

    print(len(sweep_morphing.morphing_list))
    dist = sf.sweep_frechet_compute_lower_bound(P_curve, Q_curve)
    print(dist)
    # assert False
