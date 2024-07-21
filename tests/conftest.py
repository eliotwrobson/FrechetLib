import itertools as it
from pathlib import Path

import numpy as np


def get_test_curve(curve_dir: str) -> np.ndarray:
    file_dir = Path("tests/test_data") / Path(curve_dir)

    with open(file_dir) as file:
        contents = file.read()

    lines = contents.split("\n")
    x_coords = lines[0].split()
    y_coords = lines[1].split()

    assert len(x_coords) == len(y_coords)

    num_points = len(x_coords)

    output_curve: np.ndarray = np.ndarray(shape=(num_points, 2), dtype=np.float64)

    for i, x_coord, y_coord in zip(it.count(0), x_coords, y_coords):
        output_curve[i][0] = x_coord
        output_curve[i][1] = y_coord

    return output_curve
