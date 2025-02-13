from pathlib import Path

import numpy as np


def get_test_curve(curve_dir: str) -> np.ndarray:
    file_dir = Path("tests/test_data") / Path(curve_dir)

    return np.loadtxt(file_dir, dtype=np.float64, delimiter=",")
