import numba as nb
import numba.typed as nbt
import numpy as np
from numba.experimental import jitclass
from typing_extensions import Self


@jitclass([("p_i", nb.float64[:]), ("p_j", nb.float64[:])])
class EID:
    i: int
    i_is_vert: bool
    j: int
    j_is_vert: bool

    # Computed distance between the points
    dist: float
    hash_val: int

    # Points on edges if not a vertex. Can be adjusted.
    p_i: np.ndarray
    p_j: np.ndarray

    # Parameters for the points on each curve.
    t_i: float
    t_j: float

    def __init__(
        self,
        i: int,
        i_is_vert: bool,
        j: int,
        j_is_vert: bool,
        p_i: np.ndarray,
        p_j: np.ndarray,
        t_i: float,
        t_j: float,
        dist: float,
    ) -> None:
        self.i = i
        self.i_is_vert = i_is_vert
        self.j = j
        self.j_is_vert = j_is_vert
        self.p_i = p_i
        self.p_j = p_j
        self.t_i = t_i
        self.t_j = t_j

        self.dist = dist

        self.__recompute_hash()

    def __recompute_hash(self) -> None:
        # We actually don't need p_i and p_j here since the
        # other attributes uniquely determine the point relative
        # to others
        self.hash_val = hash(
            (
                self.i,
                self.i_is_vert,
                self.j,
                self.j_is_vert,
                self.t_i,
                self.t_j,
            )
        )

    def copy(self) -> Self:
        return EID(
            self.i,
            self.i_is_vert,
            self.j,
            self.j_is_vert,
            self.p_i,
            self.p_j,
            self.t_i,
            self.t_j,
            self.dist,
        )

    def reassign_parameter_i(self, new_t: float, P: np.ndarray) -> None:
        """
        Reassign the point and parameter from the curve P.
        """
        assert 0.0 <= new_t <= 1.0

        if np.isclose(0.0, new_t):
            self.t_i = 0.0
            self.p_i = P[self.i]
        elif np.isclose(1.0, new_t):
            self.t_i = 1.0
            self.p_i = P[self.i + 1]
            # TODO maybe change the number based on index?
            # I don't think the convention matters.
        else:
            # Case where 0.0 < new_t < 1.0
            self.t_i = new_t
            self.p_i = convex_comb(P[self.i], P[self.i + 1], self.t_i)

        self.dist = float(np.linalg.norm(self.p_i - self.p_j))

        self.__recompute_hash()

    def reassign_parameter_j(self, new_t: float, Q: np.ndarray) -> None:
        """
        Reassign the point and parameter from the curve Q.
        """
        assert 0.0 <= new_t <= 1.0

        if np.isclose(0.0, new_t):
            self.t_j = 0.0
            self.p_j = Q[self.j]
        elif np.isclose(1.0, new_t):
            self.t_j = 1.0
            self.p_j = Q[self.j + 1]
            # TODO maybe change the number based on index?
            # I don't think the convention matters.
        else:
            # Case where 0.0 < new_t < 1.0
            self.t_j = new_t
            self.p_j = convex_comb(Q[self.j], Q[self.j + 1], self.t_j)

        self.dist = float(np.linalg.norm(self.p_i - self.p_j))

        self.__recompute_hash()

    def flip(self) -> None:
        self.i, self.j = self.j, self.i
        self.i_is_vert, self.j_is_vert = self.j_is_vert, self.i_is_vert
        self.p_i, self.p_j = self.p_j, self.p_i
        self.t_i, self.t_j = self.t_j, self.t_i

    def __lt__(self, other: Self) -> bool:
        return self.dist < other.dist

    def __hash__(self) -> int:
        return self.hash_val

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EID):
            return False

        return (
            (self.i == other.i)
            and (self.j == other.j)
            and (self.i_is_vert == other.i_is_vert)
            and (self.j_is_vert == other.j_is_vert)
            and np.allclose(self.p_i, other.p_i)
            and np.allclose(self.p_j, other.p_j)
            and np.isclose(self.t_i, other.t_i)
            and np.isclose(self.t_j, other.t_j)
        )


@nb.njit
def eid_get_coefficient_i(event: EID) -> float:
    return event.t_i


@nb.njit
def eid_get_coefficient_j(event: EID) -> float:
    return event.t_j


# @nb.njit
def from_coefficients(
    i: int,
    j: int,
    t_p: float,
    t_q: float,
    P: np.ndarray,
    Q: np.ndarray,
) -> EID:
    """
    Create a new EID from coefficients. This shouldn't be
    used in a VE-Frechet algorithm, since this allows for
    edge-edge matchings.
    """

    assert 0 <= i < P.shape[0]
    assert 0 <= j < Q.shape[0]

    i_is_vert = False

    # Use this to avoid issues with floating point error
    if np.isclose(t_p, 0.0):
        p_i = P[i]
        i_is_vert = True
    elif np.isclose(t_p, 1.0):
        assert i + 1 < P.shape[0]
        p_i = P[i + 1]
        i_is_vert = True
    else:
        assert i + 1 < P.shape[0]
        p_i = convex_comb(P[i], P[i + 1], t_p)

    j_is_vert = False

    # Same as above, now for Q
    if np.isclose(t_q, 0.0):
        p_j = Q[j]
        j_is_vert = True
    elif np.isclose(t_q, 1.0):
        assert j + 1 < Q.shape[0]
        p_j = Q[j + 1]
        j_is_vert = True
    else:
        assert j + 1 < Q.shape[0]
        p_j = convex_comb(Q[j], Q[j + 1], t_q)

    dist = float(np.linalg.norm(p_i - p_j))
    # TODO remove the distance from the constructor and just always
    # compute it
    return EID(i, i_is_vert, j, j_is_vert, p_i, p_j, t_p, t_q, dist)


@nb.njit
def from_curve_indices(
    i: int,
    i_is_vert: bool,
    j: int,
    j_is_vert: bool,
    P: np.ndarray,
    Q: np.ndarray,
) -> EID:
    # These values will get overwritten later
    dist = 0.0
    t_i = 0.0
    t_j = 0.0
    p_i = P[i]
    p_j = Q[j]

    assert 0 <= i < P.shape[0]
    assert 0 <= j < Q.shape[0]

    if i_is_vert and j_is_vert:
        dist = float(np.linalg.norm(P[i] - Q[j]))
    elif i_is_vert:
        if j == Q.shape[0] - 1:
            dist = float(np.linalg.norm(P[i] - Q[j]))
        else:
            dist, t_j, p_j = line_point_distance(Q[j], Q[j + 1], P[i])

    elif j_is_vert:
        if i == P.shape[0] - 1:
            dist = float(np.linalg.norm(P[i] - Q[j]))
        else:
            dist, t_i, p_i = line_point_distance(P[i], P[i + 1], Q[j])
    else:
        raise Exception

    assert 0.0 <= t_i <= 1.0
    assert 0.0 <= t_j <= 1.0

    return EID(i, i_is_vert, j, j_is_vert, p_i, p_j, t_i, t_j, dist)


@nb.njit
def get_frechet_dist_from_morphing_list(morphing_list: nb.types.ListType) -> float:
    res = 0.0

    for event in morphing_list:
        res = max(res, event.dist)

    return res


# I think this is needed at the global scope because numba has issues
# https://github.com/numba/numba/issues/7291
eid_type = nb.typeof(EID(0, True, 0, True, np.empty(0), np.empty(0), 0.0, 0.0, 0.0))


# https://numba.discourse.group/t/how-do-i-create-a-jitclass-that-takes-a-list-of-jitclass-objects/366
@jitclass(
    [
        ("morphing_list", nb.types.ListType(EID.class_type.instance_type)),
        ("P", nb.float64[:, :]),
        ("Q", nb.float64[:, :]),
    ]
)
class Morphing:
    morphing_list: nb.types.ListType
    P: np.ndarray
    Q: np.ndarray
    dist: float

    def __init__(
        self,
        morphing_list_: nb.types.ListType,
        P_: np.ndarray,
        Q_: np.ndarray,
        dist_: float,
    ):
        self.morphing_list = morphing_list_
        self.P = P_
        self.Q = Q_
        self.dist = dist_

    def flip(self) -> None:
        """
        Flips P and Q in this morphing.
        """
        for event in self.morphing_list:
            event.flip()

        self.P, self.Q = self.Q, self.P

    def copy(self) -> Self:
        new_morphing = nbt.List.empty_list(eid_type, len(self.morphing_list))

        for event in self.morphing_list:
            new_morphing.append(event.copy())

        return Morphing(new_morphing, self.P, self.Q, self.dist)

    def is_monotone(self) -> bool:
        """
        Returns True if this morphing is monotone, False otherwise.
        """

        for k in range(len(self.morphing_list) - 1):
            event = self.morphing_list[k]
            next_event = self.morphing_list[k + 1]

            if (event.i_is_vert and event.j_is_vert) or (
                next_event.i_is_vert and next_event.j_is_vert
            ):
                continue

            # Matching first or second events
            first_matches = (
                event.i == next_event.i and event.i_is_vert == next_event.i_is_vert
            )
            second_matches = (
                event.j == next_event.j and event.j_is_vert == next_event.j_is_vert
            )

            # Check tuples to see if events are on the same edge
            # and if monotonicity is violated
            if (first_matches and event.t_i > next_event.t_i) or (
                second_matches and event.t_j > next_event.t_j
            ):
                return False

        return True

    def make_monotone(self) -> None:
        """
        Modifies this morphing to be monotone in-place.
        Based on:
        https://github.com/sarielhp/FrechetDist.jl/blob/main/src/morphing.jl#L172
        """

        longest_dist = 0.0
        morphing = self.morphing_list
        n = len(morphing)
        k = 0

        while k < n:
            event = morphing[k]

            if event.i_is_vert and event.j_is_vert:
                k += 1
                continue

            elif not event.i_is_vert:
                new_k = k
                best_t = event.t_i

                while (
                    new_k < n - 1
                    and morphing[new_k + 1].i_is_vert == event.i_is_vert
                    and morphing[new_k + 1].i == event.i
                ):
                    new_event = morphing[new_k]  # .copy(morphing_obj.P, morphing_obj.Q)
                    best_t = max(best_t, new_event.t_i)
                    # TODO might be the wrong condition??
                    if best_t > new_event.t_i:
                        morphing[new_k].reassign_parameter_i(best_t, self.P)

                    longest_dist = max(longest_dist, morphing[new_k].dist)

                    new_k += 1

                new_event = morphing[new_k]  # .copy(morphing_obj.P, morphing_obj.Q)

                if best_t > new_event.t_i:
                    new_event.reassign_parameter_i(best_t, self.P)

                longest_dist = max(longest_dist, new_event.dist)
                k = new_k + 1

            # TODO might be able to simplify this?
            elif not event.j_is_vert:
                new_k = k
                best_t = event.t_j

                while (
                    new_k < n - 1
                    and morphing[new_k + 1].j_is_vert == event.j_is_vert
                    and morphing[new_k + 1].j == event.j
                ):
                    new_event = morphing[new_k]  # .copy(morphing_obj.P, morphing_obj.Q)
                    best_t = max(best_t, new_event.t_j)

                    # TODO might be the wrong condition??
                    if best_t > new_event.t_j:
                        new_event.reassign_parameter_j(best_t, self.Q)

                    longest_dist = max(longest_dist, new_event.dist)
                    # res.append(new_event)

                    new_k += 1

                new_event = morphing[new_k]  # .copy(morphing_obj.P, morphing_obj.Q)

                if best_t > new_event.t_j:
                    new_event.reassign_parameter_j(best_t, self.Q)

                longest_dist = max(longest_dist, new_event.dist)
                # res.append(new_event)
                k = new_k + 1

        self.dist = longest_dist

    def __len__(self) -> int:
        return len(self.morphing_list)

    def _print_event_list(self) -> None:
        for event in self.morphing_list:
            print(event.i, event.i_is_vert, event.j, event.j_is_vert, event.t)

    def get_prm(self) -> np.ndarray:
        # TODO once I write tests for this, it's probably possible to remove the
        # helper function and compute the prefix lengths on-the-fly. This will save
        # time / memory.
        p_lens = get_prefix_lens(self.P)
        q_lens = get_prefix_lens(self.Q)

        prm = np.empty((2, len(self.morphing_list)))

        p_events = prm[0]
        q_events = prm[1]

        n_p = p_lens.shape[0]
        n_q = q_lens.shape[0]

        # TODO Apparently sometimes it's possible to have non-zero coefficient
        # while being at the last index of the morphing. Figure out where that's
        # happening. This is why there are >= checks below
        print(n_p, n_q)
        for k in range(len(self.morphing_list)):
            event = self.morphing_list[k]
            assert 0 <= event.i < n_p
            assert 0 <= event.j < n_q
            print(p_lens[event.i], q_lens[event.j])
            # Add event to P event list
            # TODO check that this equality condition still gives you the
            # correct answer
            if event.i_is_vert or event.i + 1 >= n_p:
                p_events[k] = p_lens[event.i]
            else:
                curr_len = p_lens[event.i]
                assert event.i + 1 < n_p
                next_len = p_lens[event.i + 1]
                p_events[k] = curr_len + event.t_i * (next_len - curr_len)

            # Add event to Q event list
            if event.j_is_vert or event.j + 1 >= n_q:
                q_events[k] = q_lens[event.j]
            else:
                curr_len = q_lens[event.j]
                assert event.j + 1 < n_q
                next_len = q_lens[event.j + 1]

                # TODO switch this with convex combination helper function
                q_events[k] = curr_len + event.t_j * (next_len - curr_len)

        return prm


@nb.njit
def convex_comb(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
    return p + t * (q - p)


@nb.njit
def line_point_distance(
    p1: np.ndarray, p2: np.ndarray, q: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """
    Based on: https://stackoverflow.com/a/1501725/2923069

    Computes the point on the segment p1-p2 closest to q.
    Returns the distance between the point and the segment,
    the parameter t from p1 to p2 witnessing the point on the
    segment, and the witness point itself.

    """
    # Return minimum distance between line segment p1-p2 and point q

    q_diff = q - p1
    p_diff = p2 - p1

    l2 = np.linalg.norm(p_diff) ** 2  # i.e. |p2-p1|^2
    if np.isclose(l2, 0.0):  # p1 == p2 case
        return float(np.linalg.norm(q_diff)), 0.0, p1
    # Consider the line extending the segment, parameterized as v + t (p2 - p1).
    # We find projection of point q onto the line.
    # It falls where t = [(q-p1) . (p2-p1)] / |p2-p1|^2
    # We clamp t from [0,1] to handle points outside the segment vw.
    t = np.dot(q_diff, p_diff) / l2

    if t <= 0.0:
        return float(np.linalg.norm(q_diff)), 0.0, p1
    elif t >= 1.0:
        return float(np.linalg.norm(q - p2)), 1.0, p2

    point_on_segment = convex_comb(p1, p2, t)
    return float(np.linalg.norm(q - point_on_segment)), t, point_on_segment


@nb.njit
def get_prefix_lens(P: np.ndarray) -> np.ndarray:
    n = P.shape[0]
    prefix_lens = np.empty(n)

    curr_len = 0.0

    for i in range(n - 1):
        prefix_lens[i] = curr_len
        curr_len += np.linalg.norm(P[i] - P[i + 1])

    prefix_lens[n - 1] = curr_len

    return prefix_lens


@nb.njit
def eval_pl_func_on_dim(p: np.ndarray, q: np.ndarray, val: float, d: int) -> float:
    t = (val - p[d]) / (q[d] - p[d])
    return p * (1.0 - t) + q * t


@nb.njit
def eval_pl_func(p: np.ndarray, q: np.ndarray, val: float) -> float:
    assert p.shape == q.shape
    assert p.shape[0] == q.shape[0] == 2
    return eval_pl_func_on_dim(p, q, val, 0)[1]


@nb.njit
def eval_inv_pl_func(p: np.ndarray, q: np.ndarray, val: float) -> float:
    assert p.shape == q.shape
    assert p.shape[0] == q.shape[0] == 2
    return eval_pl_func_on_dim(p, q, val, 1)[0]


@nb.njit
def coefficient_from_prefix_lens(
    distance_along_curve: float, p_lens: np.ndarray, idx: int
) -> float:
    if idx == p_lens.shape[0] - 1:
        assert np.isclose(distance_along_curve, p_lens[idx])
        return 0.0

    edge_len = p_lens[idx + 1] - p_lens[idx]
    t = (distance_along_curve - p_lens[idx]) / edge_len

    return t


# @nb.njit
def morphing_combine(
    morphing_1: Morphing,
    morphing_2: Morphing,
) -> Morphing:
    # TODO this function only works on monotone morphings (I think). Use the
    # same helper function that Sariel used to fail when violations to the
    # monotonicity are found.
    # Code is based on:
    # https://github.com/sarielhp/FrechetDist.jl/blob/main/src/morphing.jl#L430

    # print("Starting combine")

    P = morphing_2.P
    # Original curve equal to morphing_1.P
    assert np.allclose(morphing_1.P[-1], morphing_2.Q[-1])
    R = morphing_1.Q

    prm_1 = morphing_1.get_prm()
    prm_2 = morphing_2.get_prm()

    q_events_1, r_events = prm_1
    p_events, q_events_2 = prm_2

    # print(prm_1)
    # print(prm_2)

    assert np.allclose(q_events_1[-1], q_events_2[-1])

    idx_1 = 0
    idx_2 = 0

    len_1 = q_events_1.shape[0]
    len_2 = q_events_2.shape[0]

    new_prm = []

    # P = morphing_2.P
    # Q = morphing_2.Q = morphing_1.P
    # R = morphing_1.Q
    i = 0
    while idx_1 < len_1 - 1 or idx_2 < len_2 - 1:
        print(idx_1, idx_2, q_events_1.shape[0], q_events_2.shape[0])
        q_event_1 = q_events_1[idx_1]
        q_event_2 = q_events_2[idx_2]

        is_equal = np.isclose(q_event_1, q_event_2)

        # if is_equal and idx_1 == len_1 - 1 and idx_2 == len_2 - 1:
        #     assert False
        #     new_prm.append((p_events[idx_2], r_events[idx_1]))

        if (
            is_equal
            and idx_1 < len_1 - 1
            and np.isclose(q_events_1[idx_1 + 1], q_event_1)
        ):
            # print("case2")
            new_prm.append((p_events[idx_2], r_events[idx_1]))
            idx_1 += 1

        elif (
            is_equal
            and idx_2 < len_2 - 1
            and np.isclose(q_events_2[idx_2 + 1], q_event_2)
        ):
            # print("case3")
            new_prm.append((p_events[idx_2], r_events[idx_1]))
            idx_2 += 1

        elif is_equal:
            # print("case4")
            new_prm.append((p_events[idx_2], r_events[idx_1]))
            idx_1 = min(idx_1 + 1, len_1 - 1)
            idx_2 = min(idx_2 + 1, len_2 - 1)

        # NOTE I think everything above this line is right
        # TODO Check for floating point errors
        elif q_event_1 < q_event_2:
            # print("case5")
            new_p = eval_inv_pl_func(prm_2[:, idx_2 - 1], prm_2[:, idx_2], q_event_1)
            # Enforcing monotonicity in the case of floating point error
            new_p = max(prm_2[:, idx_2 - 1][0], new_p)

            new_prm.append((new_p, r_events[idx_1]))
            idx_1 = min(idx_1 + 1, len_1 - 1)

        elif q_event_1 > q_event_2:
            # print("case6")
            new_r = eval_pl_func(prm_1[:, idx_1 - 1], prm_1[:, idx_1], q_event_2)
            # TODO double check whether or not this line is necessary
            # new_r = max(prm_1[idx_1 - 1], new_r)
            new_prm.append((p_events[idx_2], new_r))
            idx_2 = min(idx_2 + 1, len_2 - 1)

        else:
            raise Exception("Should never get here")

    q_event_1 = q_events_1[idx_1]
    q_event_2 = q_events_2[idx_2]
    assert (
        idx_1 == len_1 - 1 and idx_2 == len_2 - 1 and np.isclose(q_event_1, q_event_2)
    )

    new_prm.append((p_events[idx_2], r_events[idx_1]))

    # Now that we have the new PRM, need to extract new event
    # sequences
    new_event_sequence = nbt.List.empty_list(eid_type)
    # TODO this information gets computed when the initial PRMs are
    # created. Avoid recomputing it if possible.
    p_lens = get_prefix_lens(P)
    r_lens = get_prefix_lens(R)

    i_p = 0
    i_r = 0

    p_num_pts = p_lens.shape[0]
    r_num_pts = r_lens.shape[0]

    max_dist = 0.0
    # TODO getting to this point and seeing that neither point is a
    # vertex, so something is wrong with the PRMs. Need to debug
    # using Sariel's code.
    # NOTE Thing that is wrong is that EID events don't allow
    # for edge-edge events. Need to add a constructor that does this.

    for i in range(len(new_prm) - 1):
        p_loc, r_loc = new_prm[i]

        while i_p < p_num_pts - 1 and p_loc >= p_lens[i_p + 1]:
            i_p += 1

        assert i_p == p_num_pts - 1 or p_lens[i_p] <= p_loc

        while i_r < r_num_pts - 1 and r_loc >= r_lens[i_r + 1]:
            i_r += 1

        assert i_p == r_num_pts - 1 or r_lens[i_r] <= r_loc

        t_p = coefficient_from_prefix_lens(p_loc, p_lens, i_p)
        t_r = coefficient_from_prefix_lens(r_loc, r_lens, i_r)

        # print("about to assert")

        new_event = from_coefficients(i_p, i_r, t_p, t_r, P, R)

        max_dist = max(max_dist, new_event.dist)
        new_event_sequence.append(new_event)

    # TODO maybe add end event
    return Morphing(new_event_sequence, P, R, max_dist)


def extract_vertex_radii(
    P: np.ndarray, Q: np.ndarray, morphing: list[EID]
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each vertex in either polygon, take the maximum leash length
    on the given vertices.
    """

    P_leash_lens = np.zeros(P.shape[0], dtype=np.float64)
    Q_leash_lens = np.zeros(Q.shape[0], dtype=np.float64)

    for k in range(len(morphing)):
        event = morphing[k]
        if event.i_is_vert:
            P_leash_lens[event.i] = np.max(P_leash_lens[event.i], event.dist)

        if event.j_is_vert:
            Q_leash_lens[event.j] = np.max(Q_leash_lens[event.j], event.dist)

    return P_leash_lens, Q_leash_lens


def extract_offsets(
    P: np.ndarray, Q: np.ndarray, morphing: list[EID]
) -> tuple[np.ndarray, np.ndarray]:
    # I think this is radii without the vertex-vertex restriction
    P_offsets = np.zeros(P.shape[0], dtype=np.float64)
    Q_offsets = np.zeros(Q.shape[0], dtype=np.float64)

    for k in range(len(morphing)):
        event = morphing[k]
        P_offsets[event.i] = np.max(P_offsets[event.i], event.dist)
        Q_offsets[event.j] = np.max(Q_offsets[event.j], event.dist)

    return P_offsets, Q_offsets


@nb.njit
def simplify_polygon_radii(P: np.ndarray, r: np.ndarray) -> list[int]:
    assert P.shape[0] == r.shape

    curr = P[0]
    curr_r = r[0]
    indices = [0]
    n = P.shape[0]

    for i in range(1, n):
        curr_r = min(curr_r, r[i])
        if np.linalg.norm(P[i] - curr) > curr_r:
            curr = P[i]
            if i < n - 1:
                curr_r = r[i + 1]
            indices.append(i)

    indices.append(n - 1)

    return indices
