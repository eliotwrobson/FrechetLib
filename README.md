# FrechetLib
A library for computing Frechet distances between curves, and other associated quantities. This is
a Python version of the [FrechetDist](https://github.com/sarielhp/FrechetDist.jl) julia package.


## Development Status
This library is in an _alpha_ state. Most functions have some degree of testing, but
not extensively.

## Examples
```python
import time

from frechetlib.continuous_frechet import frechet_c_approx
from frechetlib.data import FrechetDownloader

downloader = FrechetDownloader()
big_curve_1 = downloader.get_curve("11/poly_a.txt")
big_curve_2 = downloader.get_curve("11/poly_b.txt")

print(big_curve_1.shape)
print(big_curve_2.shape)
# NOTE Run twice to factor out the compilation time from the timing.
frechet_c_approx(big_curve_1, big_curve_2, 1.01)
print("Starting")

start = time.perf_counter()
res, morphing = frechet_c_approx(big_curve_1, big_curve_2, 1.01)
end = time.perf_counter()

print("Approximated Frechet distance: ", morphing.dist)
print("Time taken (s): ", end - start)
```

## Benchmarks

To run the standard benchmarks, you first need to have [`poetry`](https://python-poetry.org/) installed.
After cloning the repo, run the command 
```sh
poetry install --with dev
```
That will install the package from source in a virtual environment with the dev dependencies.
To start the benchmarks, run
```sh
poetry run python paper_bench/run_experiments.py
```
This will then execute the benchmarks and put the results in `benchmark_results.csv`.

## API

The package API is currently minimal and undocumented for now. See the test cases for usage
examples. This will be filled out more later.
