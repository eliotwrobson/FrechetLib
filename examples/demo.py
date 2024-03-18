import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from frechetlib.continuous_frechet import frechet_mono_via_refinement
from matplotlib.animation import FuncAnimation


def generate_time_series(
    n: int, d: int, low: float, high: float, drift: None | float = None
) -> np.ndarray:
    if drift is None:
        drift = (high - low) / 100.0

    midpoint = (high + low) / 2.0

    res: np.ndarray = np.ndarray((n, d))
    res[0] = np.random.uniform(low=midpoint, high=midpoint, size=d)
    res[0][0] = 0.0

    for i in range(1, n):
        noise_vec = np.random.uniform(low=-drift, high=drift, size=d)
        res[i] = res[i - 1] + noise_vec
        res[i][0] = float(i)

    return res


def main() -> None:
    np.random.seed(12345)
    num_pts = 100
    d = 2
    noise_limit = 1.0
    approx = 1.01

    low = 0.0
    high = 100.0
    midpoint = (high - low) / 2
    drift = (high - low) / num_pts

    P = generate_time_series(num_pts, d, low=low, high=high, drift=drift)
    Q = P + np.random.uniform(low=-noise_limit, high=noise_limit, size=(num_pts, d))

    # Don't time the first run so the compilation step runs first
    frechet_mono_via_refinement(P, Q, approx)

    start = time.perf_counter()
    P, Q, morphing, dist, is_exact = frechet_mono_via_refinement(P, Q, approx)
    end = time.perf_counter()

    print("Compute time: ", end - start)

    # Taken from https://matplotlib.org/stable/api/animation_api.html
    fig, ax = plt.subplots()
    fig.set_size_inches(13, 8)
    xdata1, ydata1 = [], []
    xdata2, ydata2 = [], []
    (line1,) = ax.plot([], [], lw=2)
    (line2,) = ax.plot([], [], lw=2)
    (leash,) = ax.plot([], [], lw=2, color="red")

    for morphing_frame in morphing:
        if morphing_frame.i_is_vert and morphing_frame.j_is_vert:
            p_point = P[morphing_frame.i]
            q_point = Q[morphing_frame.j]
        elif morphing_frame.i_is_vert:
            p_point = P[morphing_frame.i]
            q_point = morphing_frame.p
        elif morphing_frame.j_is_vert:
            q_point = Q[morphing_frame.j]
            p_point = morphing_frame.p

        xdata1.append(p_point[0])
        ydata1.append(p_point[1])

        xdata2.append(q_point[0])
        ydata2.append(q_point[1])

    def init():
        ax.set_xlim(0.0, num_pts)
        ax.set_ylim(midpoint - 20, midpoint + 20)
        return (line1,)

    def update(morphing_frame):
        line1.set_data(xdata1, ydata1)

        line2.set_data(xdata2, ydata2)

        if morphing_frame.i_is_vert and morphing_frame.j_is_vert:
            p_point = P[morphing_frame.i]
            q_point = Q[morphing_frame.j]
        elif morphing_frame.i_is_vert:
            p_point = P[morphing_frame.i]
            q_point = morphing_frame.p
        elif morphing_frame.j_is_vert:
            q_point = Q[morphing_frame.j]
            p_point = morphing_frame.p

        leash.set_data(
            [p_point[0], q_point[0]],
            [p_point[1], q_point[1]],
        )
        return (line1, line2, leash)

    ani = FuncAnimation(fig, update, frames=morphing, init_func=init, blit=True)

    writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save("frechet_monotone.gif", writer=writer)

    plt.show()


if __name__ == "__main__":
    main()
