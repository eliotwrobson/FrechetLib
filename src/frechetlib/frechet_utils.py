import numba as nb
import numpy as np

from frechetlib.retractable_frechet import EID


@nb.njit
def line_point_distance(
    p1: np.ndarray, p2: np.ndarray, q: np.ndarray
) -> tuple[float, float]:
    """
    Based on: https://stackoverflow.com/a/1501725/2923069
    """
    # Return minimum distance between line segment p1-p2 and point q
    l2 = np.linalg.norm(p1 - p2)  # i.e. |p2-p1|^2 -  avoid a sqrt
    if np.isclose(l2, 0.0):  # p1 == p2 case
        return float(np.linalg.norm(q - p1)), 0.0
    # Consider the line extending the segment, parameterized as v + t (p2 - p1).
    # We find projection of point q onto the line.
    # It falls where t = [(q-p1) . (p2-p1)] / |p2-p1|^2
    # We clamp t from [0,1] to handle points outside the segment vw.
    t = np.dot(q - p1, p2 - p1) / l2

    if t <= 0.0:
        return float(np.linalg.norm(p1 - q)), t
    elif t >= 1.0:
        return float(np.linalg.norm(p2 - q)), t

    return float(np.linalg.norm(q - (p1 + t * (p2 - p1)))), t


@nb.njit
def get_prefix_lens(P: np.ndarray) -> np.ndarray:
    prefix_lens = []

    curr_len = 0.0

    for i in range(len(P) - 1):
        prefix_lens.append(curr_len)
        curr_len += np.linalg.norm(P[i] - P[i + 1])

    prefix_lens.append(curr_len)

    return np.ndarray(prefix_lens)


@nb.njit
def morphing_get_prm(
    P: np.ndarray, Q: np.ndarray, events: list[EID]
) -> tuple[list[float], list[float]]:
    p_lens = get_prefix_lens(P)
    q_lens = get_prefix_lens(Q)

    p_events = []
    q_events = []
    for k in range(len(events)):
        event = events[k]
        # Add event to P event list
        if event.i_is_vert:
            p_events.append(p_lens[event.i])
        else:
            curr_len = p_lens[event.i]
            next_len = p_lens[event.i + 1]
            p_events.append(curr_len + event.t * (next_len - curr_len))

        # Add event to Q event list
        if event.j_is_vert:
            p_events.append(q_lens[event.j])
        else:
            curr_len = q_lens[event.j]
            next_len = q_lens[event.j + 1]
            p_events.append(curr_len + event.t * (next_len - curr_len))

    return p_events, q_events


@nb.njit
def eval_pl_func_on_dim(p: np.ndarray, q: np.ndarray, val: float, d: int) -> float:
    t = (val - p[d]) / (q[d] - p[d])
    return p * (1.0 - t) + q * t


@nb.njit
def eval_pl_func(p: np.ndarray, q: np.ndarray, val: float) -> float:
    return eval_pl_func(p, q, val, 0)[1]


@nb.njit
def eval_inv_pl_func(p: np.ndarray, q: np.ndarray, val: float) -> float:
    return eval_pl_func(p, q, val, 1)[0]


def morphing_combine(
    P: np.ndarray,
    Q: np.ndarray,
    U: np.ndarray,
    morphing_1: list[EID],
    morphing_2: list[EID],
) -> list[EID]:
    prm_1 = morphing_get_prm(morphing_1)
    prm_2 = morphing_get_prm(morphing_2)

    # Apparently len(prm_1[1]) == len(prm_2[0])

    idx_1 = 0
    idx_2 = 0

    len_1 = len(prm_1)
    len_2 = len(prm_2)

    res = []

    while idx_1 < len_1 or idx_2 < len_2:
        # x = dim(g,1)
        # y = dim(g,2) = dim(f, 1)
        # z = dim(f,2)
        x, y_1 = prm_1[idx_1]
        y_2, z = prm_2[idx_2]

        is_equal = np.isclose(y_1, y_2)

        if is_equal and idx_1 == len_1 and idx_2 == len_2:
            res.append((y_1, y_2))
        elif is_equal and idx_1 < len_1 and np.isclose(prm_1[idx_1 + 1][1], y_1):
            res.append((y_1, y_2))
            idx_1 += 1
        elif is_equal and idx_2 < len_2 and np.isclose(prm_2[idx_2 + 1][0], y_2):
            res.append((y_1, y_2))
            idx_2 += 1
        elif is_equal:
            res.append((y_1, y_2))
            idx_1 = min(idx_1 + 1, len_1)
            idx_2 = min(idx_2 + 1, len_2)
        elif y_1 < y_2:
            new_x = eval_inv_pl_func(prm_1[idx_1 - 1], prm_1[idx_1], y_2)
            new_x = max(prm_1[idx_1 - 1], new_x)
            res.append(new_x, z)
            idx_2 = min(idx_2 + 1, len_2)
        elif y_1 > y_2:
            new_z = eval_pl_func(prm_2[idx_2 - 1], prm_2[idx_2], y_1)
            new_z = max(prm_2[idx_2 - 1], new_z)
            res.append(x, new_z)
            idx_1 = min(idx_1 + 1, len_1)

    return res
