import numba as nb
import numpy as np
from numba.experimental import jitclass
from typing_extensions import Self


@jitclass
class EID:
    i: int
    i_is_vert: bool
    j: int
    j_is_vert: bool
    dist: float
    t: float

    def __init__(
        self,
        i_: int,
        i_is_vert_: bool,
        j_: int,
        j_is_vert_: bool,
        P: np.ndarray,
        Q: np.ndarray,
    ) -> None:
        self.i = i_
        self.i_is_vert = i_is_vert_
        self.j = j_
        self.j_is_vert = j_is_vert_
        self.t = 0.0

        # Compute the distance
        if self.i_is_vert:
            if self.j_is_vert:
                self.dist = float(np.linalg.norm(P[self.i] - Q[self.j]))

            else:
                self.dist, self.t = line_point_distance(
                    Q[self.j], Q[self.j + 1], P[self.i]
                )

        elif self.j_is_vert:
            self.dist, self.t = line_point_distance(P[self.i], P[self.i + 1], Q[self.j])
        else:
            raise Exception

    def __lt__(self, other: Self) -> bool:
        return self.dist < other.dist

    def __hash__(self) -> int:
        return hash((self.i, self.i_is_vert, self.j, self.j_is_vert))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EID):
            return False

        return (
            (self.i == other.i)
            and (self.j == other.j)
            and (self.i_is_vert == other.i_is_vert)
            and (self.j_is_vert == other.j_is_vert)
        )


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
def convex_comb(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
    return p * (1.0 - t) + q * t


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


@nb.njit
def morphing_combine(
    P: np.ndarray,
    Q: np.ndarray,
    U: np.ndarray,
    morphing_1: list[EID],
    morphing_2: list[EID],
) -> list[EID]:
    prm_1 = morphing_get_prm(P, Q, morphing_1)
    prm_2 = morphing_get_prm(Q, U, morphing_2)

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
