import itertools as it
import typing as t

import numpy as np
import numpy.typing as npt
import pytest
import utils as u

import frechetlib.discrete_frechet as df

FrechetDistFuncT = t.Callable[[np.ndarray, np.ndarray], df._DiscreteReturnT]
CurveGeneratorFunctionT = t.Callable[[int, float], tuple[np.ndarray, np.ndarray]]


DISCRETE_FRECHET_FUNCS = (df.linear_frechet, df.linear_frechet_2)

# Start of actual test functions


@pytest.mark.parametrize(
    "generate_curves", [u.generate_curves_random, u.generate_curves_close]
)
def test_frechet_equal(generate_curves: CurveGeneratorFunctionT) -> None:
    n = 1000
    P, Q = generate_curves(n, 100.0)
    assert np.isclose(df.linear_frechet(P, Q)[0], df.linear_frechet_2(P, Q)[0])


@pytest.mark.parametrize(
    "frechet_dist_func",
    DISCRETE_FRECHET_FUNCS,
)
@pytest.mark.parametrize(
    "generate_curves", [u.generate_curves_random, u.generate_curves_close]
)
@pytest.mark.parametrize("n", [100, 1000])
def test_frechet_benchmark_morphing(
    benchmark: t.Any,
    frechet_dist_func: FrechetDistFuncT,
    generate_curves: CurveGeneratorFunctionT,
    n: int,
) -> None:
    P, Q = generate_curves(n, 100.0)
    dist, morphing = benchmark(frechet_dist_func, P, Q)

    # Check that the morphing is valid
    i, j = morphing[0]
    assert u.leq_with_tolerance(dist, np.linalg.norm(P[i] - Q[j]))

    for (i1, j1), (i2, j2) in it.pairwise(morphing):
        # Check that change is legal
        assert (i2 - i1, j2 - j1) in {(0, 1), (1, 0), (1, 1)}

        # Check the given frechet distance is respected by the morphing
        assert u.leq_with_tolerance(dist, np.linalg.norm(P[i2] - Q[j2]))


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

    assert frechet_dist_func(P, Q)[0] == expected


@pytest.mark.parametrize(
    "frechet_dist_func",
    DISCRETE_FRECHET_FUNCS,
)
def test_errors(frechet_dist_func: FrechetDistFuncT) -> None:
    P = np.array([])
    Q = np.array([[2, 2], [0, 1], [2, 4]])

    with pytest.raises(ValueError):
        assert frechet_dist_func(P, Q)[0] == 2.0
