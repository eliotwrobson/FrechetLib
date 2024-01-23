import typing as t

import numpy as np
import numpy.typing as npt
import pytest

import frechetlib.discrete_frechet as df

FrechetDistFuncT = t.Callable[[np.ndarray, np.ndarray], np.float64]
CurveGeneratorFunctionT = t.Callable[[int, float], tuple[np.ndarray, np.ndarray]]


DISCRETE_FRECHET_FUNCS = (df.linear_frechet, df.linear_frechet_2)
# Generator helper functions


def generate_curves_random(
    num_pts: int, scaling_factor: float
) -> tuple[np.ndarray, np.ndarray]:
    P = np.random.rand(num_pts, 2) * scaling_factor
    Q = np.random.rand(num_pts, 2) * scaling_factor
    return (P, Q)


def generate_curves_close(
    num_pts: int, scaling_factor: float
) -> tuple[np.ndarray, np.ndarray]:
    P = np.random.uniform(low=1.0, high=scaling_factor, size=(num_pts, 2))
    Q = P + np.random.uniform(low=0.0, high=0.5, size=(num_pts, 2))
    return (P, Q)


# Start of actual test functions


@pytest.mark.parametrize(
    "generate_curves", [generate_curves_random, generate_curves_close]
)
def test_frechet_equal(generate_curves: CurveGeneratorFunctionT) -> None:
    n = 1000
    P, Q = generate_curves(n, 100.0)
    assert np.isclose(df.linear_frechet(P, Q), df.linear_frechet_2(P, Q))


@pytest.mark.parametrize(
    "frechet_dist_func",
    DISCRETE_FRECHET_FUNCS,
)
@pytest.mark.parametrize(
    "generate_curves", [generate_curves_random, generate_curves_close]
)
@pytest.mark.parametrize("n", [100, 1000])
def test_frechet_benchmark(
    benchmark: t.Any,
    frechet_dist_func: FrechetDistFuncT,
    generate_curves: CurveGeneratorFunctionT,
    n: int,
) -> None:
    P, Q = generate_curves(n, 100.0)
    benchmark(frechet_dist_func, P, Q)


@pytest.mark.parametrize(
    "frechet_dist_func",
    DISCRETE_FRECHET_FUNCS,
)
@pytest.mark.parametrize(
    "P, Q, expected",
    [
        ([[1, 1], [2, 1], [2, 2]], [[2, 2], [0, 1], [2, 4]], 2.0),
        (
            np.array((np.linspace(0.0, 1.0, 100), np.ones(100))).T,
            np.array((np.linspace(0.0, 1.0, 100), np.ones(100))).T,
            0,
        ),
        ([[-1, 0], [0, 1], [1, 0], [0, -1]], [[-2, 0], [0, 2], [2, 0], [0, -2]], 1.0),
        (
            np.array((np.linspace(0.0, 1.0, 100), np.ones(100) * 2)).T,
            np.array((np.linspace(0.0, 1.0, 100), np.ones(100))).T,
            1.0,
        ),
        ([[1, 1], [2, 1]], [[2, 2], [0, 1], [2, 4]], 3.0),
        (
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.1], [2.0, 1.2], [3.0, 1.1], [4.0, 1.0]],
            1.2,
        ),
    ],
)
def test_discrete_frechet(
    frechet_dist_func: FrechetDistFuncT,
    P: npt.ArrayLike,
    Q: npt.ArrayLike,
    expected: float,
) -> None:
    P = np.array(P, np.float64)
    Q = np.array(Q, np.float64)

    assert frechet_dist_func(P, Q) == expected


@pytest.mark.parametrize(
    "frechet_dist_func",
    DISCRETE_FRECHET_FUNCS,
)
def test_errors(frechet_dist_func: FrechetDistFuncT) -> None:
    P = np.array([])
    Q = np.array([[2, 2], [0, 1], [2, 4]])

    with pytest.raises(ValueError):
        assert frechet_dist_func(P, Q) == 2.0
