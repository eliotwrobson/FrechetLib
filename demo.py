import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from frechetlib.retractable_frechet import retractable_ve_frechet


def generate_time_series(
    n: int, d: int, low: float, high: float, drift: None | float = None
) -> np.ndarray:
    if drift is None:
        drift = (high - low) / 100.0

    midpoint = (high + low) / 2.0

    res = np.ndarray((n, d))
    res[0] = np.random.uniform(low=midpoint, high=midpoint, size=d)
    res[0][0] = 0.0

    for i in range(1, n - 1):
        noise_vec = np.random.uniform(low=0.0, high=drift, size=d)
        res[i] = res[i - 1] + noise_vec
        res[i][0] = float(i)

    return res


def main() -> None:
    num_pts = 50
    d = 2
    noise_limit = 10.0

    low = 0.0
    high = 100.0

    P = generate_time_series(num_pts, d, low=low, high=high)
    Q = P + np.random.uniform(low=0.0, high=noise_limit, size=(num_pts, d))

    # Don't time the first run so the compilation step runs first
    retractable_ve_frechet(P, Q)

    start = time.perf_counter()
    dist, morphing = retractable_ve_frechet(P, Q)
    end = time.perf_counter()

    print("Compute time: ", end - start)

    # Taken from https://matplotlib.org/stable/api/animation_api.html
    fig, ax = plt.subplots()
    xdata1, ydata1 = [], []
    xdata2, ydata2 = [], []
    (line1,) = ax.plot([], [], lw=2)
    (line2,) = ax.plot([], [], lw=2)
    (leash,) = ax.plot([], [], lw=2)

    def init():
        ax.set_xlim(0.0, num_pts)
        ax.set_ylim(low, high)
        return (line1,)

    def update(morphing_frame):
        # print(morphing_frame.i)
        xdata1.append(P[morphing_frame.i][0])
        ydata1.append(P[morphing_frame.i][1])
        line1.set_data(xdata1, ydata1)

        xdata2.append(Q[morphing_frame.j][0])
        ydata2.append(Q[morphing_frame.j][1])
        line2.set_data(xdata2, ydata2)

        leash.set_data(
            [Q[morphing_frame.j][0], P[morphing_frame.j][0]],
            [Q[morphing_frame.j][1], P[morphing_frame.j][1]],
        )
        return (line1, line2, leash)

    ani = FuncAnimation(fig, update, frames=morphing, init_func=init, blit=True)
    plt.show()


if __name__ == "__main__":
    main()
