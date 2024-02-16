import time

import matplotlib.pyplot as plt
import numpy as np

from frechetlib.retractable_frechet import retractable_ve_frechet


def main() -> None:
    num_pts = 100
    scaling_factor = 100.0
    noise_limit = 1.0
    P = np.random.uniform(low=noise_limit, high=scaling_factor, size=(num_pts, 2))
    Q = P + np.random.uniform(low=0.0, high=noise_limit, size=(num_pts, 2))

    # Don't time the first run so the compilation step runs first
    retractable_ve_frechet(P, Q)

    start = time.perf_counter()
    dist, morphing = retractable_ve_frechet(P, Q)
    end = time.perf_counter()

    print("Compute time: ", end - start)


if __name__ == "__main__":
    main()
