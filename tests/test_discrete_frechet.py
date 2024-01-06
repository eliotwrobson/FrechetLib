import numpy as np
import pytest

from frechetlib.discrete_frechet import linear_frechet

TEST_CASES = [
    {"P": [[1, 1], [2, 1], [2, 2]], "Q": [[2, 2], [0, 1], [2, 4]], "expected": 2.0},
    {
        "P": np.array((np.linspace(0.0, 1.0, 100), np.ones(100))).T,
        "Q": np.array((np.linspace(0.0, 1.0, 100), np.ones(100))).T,
        "expected": 0,
    },
    {
        "P": [[-1, 0], [0, 1], [1, 0], [0, -1]],
        "Q": [[-2, 0], [0, 2], [2, 0], [0, -2]],
        "expected": 1.0,
    },
    {
        "P": np.array((np.linspace(0.0, 1.0, 100), np.ones(100) * 2)).T,
        "Q": np.array((np.linspace(0.0, 1.0, 100), np.ones(100))).T,
        "expected": 1.0,
    },
    {"P": [[1, 1], [2, 1]], "Q": [[2, 2], [0, 1], [2, 4]], "expected": 3.0},
    {
        "P": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
        "Q": [[0.0, 1.0], [1.0, 1.1], [2.0, 1.2], [3.0, 1.1], [4.0, 1.0]],
        "expected": 1.2,
    },
]


def test_discrete_frechet() -> None:
    for test_case in TEST_CASES:
        P = np.array(test_case["P"], np.float64)
        Q = np.array(test_case["Q"], np.float64)
        eo = test_case["expected"]

        assert linear_frechet(P, Q) == eo


def test_errors() -> None:
    P = []
    Q = [[2, 2], [0, 1], [2, 4]]

    with pytest.raises(ValueError):
        assert linear_frechet(P, Q) == 2.0
