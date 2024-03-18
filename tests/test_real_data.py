import frechetlib.data as fld
import frechetlib.discrete_frechet as df
import numpy as np
import pytest


@pytest.fixture
def frechet_downloader() -> fld.FrechetDownloader:
    return fld.FrechetDownloader()


# TODO set this in a types file or something like that
# def get_curve_from_data_file(file_dir: str) -> fu.Curve:
#     with open(file_dir) as file:
#         contents = file.read()

#     lines = contents.split("\n")
#     x_coords = lines[0].split()
#     y_coords = lines[1].split()

#     assert len(x_coords) == len(y_coords)

#     num_points = len(x_coords)

#     output_curve: fu.Curve = np.ndarray(shape=(num_points, 2), dtype=np.float64)

#     for i, x_coord, y_coord in zip(it.count(0), x_coords, y_coords):
#         output_curve[i][0] = x_coord
#         output_curve[i][1] = y_coord

#     return output_curve


# TODO split this or something
# @pytest.fixture
# def curve_12() -> t.Tuple[fu.Curve, fu.Curve]:
#     # curve_P_file = pooch.retrieve(
#     #    url=R"https://sarielhp.org/p/24/frechet_ve/examples/12/poly_a.txt",
#     #    known_hash="c99e208e48275a894b0f28d503f98c9e176e96a6bffbac9ed861974390e7efd3",
#     # )
#     # curve_Q_file = pooch.retrieve(
#     #    url=R"https://sarielhp.org/p/24/frechet_ve/examples/12/poly_b.txt",
#     #    known_hash="4b11cc56991b697b07957585a6a2ff2c96ce82df401cfeb5b94aba03c7d32349",
#     # )

#     P_curve = get_curve_from_data_file(retriever("12/poly_a.txt"))
#     Q_curve = get_curve_from_data_file(retriever("12/poly_b.txt"))

#     return P_curve, Q_curve


def test_live(frechet_downloader: fld.FrechetDownloader) -> None:
    P_curve = frechet_downloader.get_curve("12/poly_a.txt")
    Q_curve = frechet_downloader.get_curve("12/poly_b.txt")
    dist, morphing = df.discrete_retractable_frechet(P_curve, Q_curve)

    assert np.isclose(dist, 0.0002966043155438105)
