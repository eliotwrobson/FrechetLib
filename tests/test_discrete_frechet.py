import typing as t

import numpy as np
import pytest

import frechetlib.discrete_frechet as df

DISCRETE_FRECHET_FUNCS = (df.linear_frechet, df.linear_frechet_2)
FrechetDistFuncT = t.Callable[[np.ndarray, np.ndarray], np.float64]


def test_frechet_equal() -> None:
    n = 1000
    P = np.random.rand(n, 2)
    Q = np.random.rand(n, 2)

    assert np.isclose(df.linear_frechet(P, Q), df.linear_frechet_2(P, Q))


@pytest.mark.parametrize(
    "frechet_dist_func",
    DISCRETE_FRECHET_FUNCS,
)
def test_frechet_benchmark_random(
    benchmark: t.Any, frechet_dist_func: FrechetDistFuncT
) -> None:
    n = 500
    P = np.random.rand(n, 2)
    Q = np.random.rand(n, 2)

    benchmark(frechet_dist_func, P, Q)


@pytest.mark.parametrize(
    "frechet_dist_func",
    DISCRETE_FRECHET_FUNCS,
)
def test_frechet_benchmark_close(
    benchmark: t.Any, frechet_dist_func: FrechetDistFuncT
) -> None:
    n = 500
    P = np.random.rand(n, 2)
    Q = P + np.random.uniform(low=0.1, high=10.0, size=(n, 2))

    benchmark(frechet_dist_func, P, Q)


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


@pytest.mark.parametrize(
    "frechet_dist_func",
    DISCRETE_FRECHET_FUNCS,
)
def test_discrete_frechet(frechet_dist_func: FrechetDistFuncT) -> None:
    for test_case in TEST_CASES:
        P = np.array(test_case["P"], np.float64)
        Q = np.array(test_case["Q"], np.float64)
        eo = test_case["expected"]

        assert frechet_dist_func(P, Q) == eo


@pytest.mark.parametrize(
    "frechet_dist_func",
    DISCRETE_FRECHET_FUNCS,
)
def test_errors(frechet_dist_func: FrechetDistFuncT) -> None:
    P = np.array([])
    Q = np.array([[2, 2], [0, 1], [2, 4]])

    with pytest.raises(ValueError):
        assert frechet_dist_func(P, Q) == 2.0
