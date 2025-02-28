from __future__ import annotations

import typing as t

import numpy as np
cimport libc.stdio
cimport numpy as cnp
from .geometry_utils cimport EID, Point, from_curve_indices, numpy_to_point_list

cdef class Curve:
    def __cinit__(self, cnp.ndarray[double, ndim=2] P):
        self.point_list = []

        cdef int n = P.shape[0]
        cdef int i
        for i in range(n):
            self.point_list.append(Point(P[i]))

cdef class Morphing:
    #morphing_list: t.List[EID]
    #P: np.ndarray
    #Q: np.ndarray
    #dist: float

    def __cinit__(
        self,
        morphing_list_: t.List[EID],
        P_: list,
        Q_: list,
        dist_: float,
    ):
        self.morphing_list = morphing_list_

        self.P = P_
        self.Q = Q_
        self.dist = dist_

    def get_dist(self) -> float:
        return self.dist

    def get_morphing_list(self) -> t.List[EID]:
        return self.morphing_list

    def get_P(self) -> np.ndarray:
        return self.P

    def get_Q(self) -> np.ndarray:
        return self.Q

    cpdef flip(self):
        """
        Flips P and Q in this morphing.
        """
        for event in self.morphing_list:
            event.flip()

        self.P, self.Q = self.Q, self.P

    cpdef Morphing copy(self):
        new_morphing = []

        for event in self.morphing_list:
            new_morphing.append(event.copy())

        return Morphing(new_morphing, self.P, self.Q, self.dist)

    cpdef bint is_monotone(self):
        """
        Returns True if this morphing is monotone, False otherwise.
        """

        for k in range(len(self.morphing_list) - 1):
            event = self.morphing_list[k]
            next_event = self.morphing_list[k + 1]

            # First, assert monotonicity on the "P" side.
            if event.get_i() > next_event.get_i():
                # print("Case 1")
                # print(
                #     event.i,
                #     event.i_is_vert,
                #     event.j,
                #     event.j_is_vert,
                #     event.t_i,
                #     event.t_j,
                # )
                # print(
                #     next_event.i,
                #     next_event.i_is_vert,
                #     next_event.j,
                #     next_event.j_is_vert,
                #     next_event.t_i,
                #     next_event.t_j,
                # )
                return False

            # TODO change checks to account for floating point issues.
            # Make it so that make_monotone gets rid of the need for this
            if event.get_i() == next_event.get_i() and event.get_t_i() > next_event.get_t_i() * 1.001:
                # print("Case 2", event.t_i, next_event.t_i)
                # print(event.i_is_vert, next_event.i_is_vert)
                # print(
                #     event.i,
                #     event.i_is_vert,
                #     event.j,
                #     event.j_is_vert,
                #     event.t_i,
                #     event.t_j,
                # )
                # print(
                #     next_event.i,
                #     next_event.i_is_vert,
                #     next_event.j,
                #     next_event.j_is_vert,
                #     next_event.t_i,
                #     next_event.t_j,
                # )
                return False

            # Next, assert monotonicity on the "Q" side.
            if event.get_j() > next_event.get_j():
                # print("Case 3")
                # print(
                #     event.i,
                #     event.i_is_vert,
                #     event.j,
                #     event.j_is_vert,
                #     event.t_i,
                #     event.t_j,
                # )
                # print(
                #     next_event.i,
                #     next_event.i_is_vert,
                #     next_event.j,
                #     next_event.j_is_vert,
                #     next_event.t_i,
                #     next_event.t_j,
                # )
                return False

            # TODO change checks to account for floating point issues.
            if event.get_j() == next_event.get_j() and event.get_t_j() > next_event.get_t_j() * 1.001:
                # print("Case 4", event.t_j, next_event.t_j)
                # print(
                #     event.i,
                #     event.i_is_vert,
                #     event.j,
                #     event.j_is_vert,
                #     event.t_i,
                #     event.t_j,
                # )
                # print(
                #     next_event.i,
                #     next_event.i_is_vert,
                #     next_event.j,
                #     next_event.j_is_vert,
                #     next_event.t_i,
                #     next_event.t_j,
                # )
                return False

        return True

    cpdef float make_monotone(self):
        """
        Modifies this morphing to be monotone in-place.
        Based on:
        https://github.com/sarielhp/FrechetDist.jl/blob/main/src/morphing.jl#L172
        """

        # TODO distance adjustment needs to be fixed for summed case.

        cdef float longest_dist = 0.0
        cdef list morphing = self.morphing_list
        cdef int n = len(morphing)
        cdef int k = 0

        # Error incurred during monotonization.
        # NOTE should be 0.0 if there is no error
        cdef float err = 0.0

        while k < n:
            event = morphing[k]

            # print(event.dist)
            if event.get_i_is_vert() and event.get_j_is_vert():
                k += 1
                longest_dist = max(longest_dist, event.get_dist())
                continue

            elif not event.get_i_is_vert():
                new_k = k
                best_t = event.get_t_i()

                while (
                    new_k < n - 1
                    and morphing[new_k + 1].get_i_is_vert() == event.get_i_is_vert()
                    and morphing[new_k + 1].get_i() == event.get_i()
                ):
                    new_event = morphing[new_k]  # .copy(morphing_obj.P, morphing_obj.Q)

                    best_t = max(best_t, new_event.get_t_i())
                    # TODO might be the wrong condition??

                    if best_t > new_event.get_t_i():
                        new_err = morphing[new_k].reassign_parameter_i(best_t, self.P)
                        err = max(err, new_err)

                    longest_dist = max(longest_dist, morphing[new_k].get_dist())

                    new_k += 1

                new_event = morphing[new_k]

                if best_t > new_event.get_t_i():
                    new_err = new_event.reassign_parameter_i(best_t, self.P)
                    err = max(err, new_err)

                longest_dist = max(longest_dist, new_event.get_dist())
                k = new_k + 1

            # TODO might be able to simplify this?
            elif not event.get_j_is_vert():
                new_k = k
                best_t = event.get_t_j()
                while (
                    new_k < n - 1
                    and morphing[new_k + 1].get_j_is_vert() == event.get_j_is_vert()
                    and morphing[new_k + 1].get_j() == event.get_j()
                ):
                    new_event = morphing[new_k]  # .copy(morphing_obj.P, morphing_obj.Q)
                    best_t = max(best_t, new_event.get_t_j())

                    # TODO might be the wrong condition??
                    if best_t > new_event.get_t_j():
                        new_err = new_event.reassign_parameter_j(best_t, self.Q)
                        err = max(err, new_err)

                    longest_dist = max(longest_dist, new_event.get_dist())

                    new_k += 1

                new_event = morphing[new_k]  # .copy(morphing_obj.P, morphing_obj.Q)

                if best_t > new_event.get_t_j():
                    new_err = new_event.reassign_parameter_j(best_t, self.Q)
                    err = max(err, new_err)

                longest_dist = max(longest_dist, new_event.get_dist())
                k = new_k + 1

        self.dist = longest_dist

        return err

    def __len__(self) -> int:
        return len(self.morphing_list)

    cpdef cnp.ndarray get_prm(self):
        # TODO once I write tests for this, it's probably possible to remove the
        # helper function and compute the prefix lengths on-the-fly. This will save
        # time / memory.
        cdef cnp.ndarray[cnp.float64_t, ndim=1] p_lens = get_prefix_lens(self.P)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] q_lens = get_prefix_lens(self.Q)

        cdef cnp.ndarray[cnp.float64_t, ndim=2] prm = np.empty((2, len(self.morphing_list)))

        cdef cnp.ndarray[cnp.float64_t, ndim=1] p_events = prm[0]
        cdef cnp.ndarray[cnp.float64_t, ndim=1] q_events = prm[1]

        cdef int n_p = p_lens.shape[0]
        cdef int n_q = q_lens.shape[0]

        cdef int k
        cdef EID event

        # TODO Apparently sometimes it's possible to have non-zero coefficient
        # while being at the last index of the morphing. Figure out where that's
        # happening. This is why there are >= checks below
        # print(n_p, n_q)
        for k in range(len(self.morphing_list)):
            event = self.morphing_list[k]
            assert 0 <= event.get_i() < n_p
            assert 0 <= event.get_j() < n_q

            # Add event to P event list
            # TODO check that this equality condition still gives you the
            # correct answer
            if event.get_i_is_vert() or event.get_i() + 1 >= n_p:
                p_events[k] = p_lens[event.get_i()]
            else:
                curr_len = p_lens[event.get_i()]
                assert event.get_i() + 1 < n_p
                next_len = p_lens[event.get_i() + 1]
                p_events[k] = curr_len + event.get_t_i() * (next_len - curr_len)

            # Add event to Q event list
            if event.get_j_is_vert() or event.get_j() + 1 >= n_q:
                q_events[k] = q_lens[event.get_j()]
            else:
                curr_len = q_lens[event.get_j()]
                assert event.get_j() + 1 < n_q
                next_len = q_lens[event.get_j() + 1]

                # TODO switch this with convex combination helper function
                q_events[k] = curr_len + event.get_t_j() * (next_len - curr_len)

            # print(p_events[k], q_events[k])
            # print()
        return prm

    def extract_vertex_radii(self) -> tuple[np.ndarray, np.ndarray]:
        """
        For each vertex in either polygon, take the maximum leash length
        on the given vertices.
        """

        P_leash_lens = np.zeros(self.P.shape[0], dtype=np.float64)
        Q_leash_lens = np.zeros(self.Q.shape[0], dtype=np.float64)

        for k in range(len(self.morphing_list)):
            event = self.morphing_list[k]
            if event.get_i_is_vert():
                P_leash_lens[event.get_i()] = max(P_leash_lens[event.get_i()], event.get_dist())  # type: ignore

            if event.get_j_is_vert():
                Q_leash_lens[event.get_j()] = max(Q_leash_lens[event.get_j()], event.get_dist())  # type: ignore

        return P_leash_lens, Q_leash_lens

########################### End of Morphing class definition ###########################


# @njit(cache=True)
cpdef cnp.ndarray convex_comb(
    cnp.ndarray[cnp.float64_t, ndim=1] p,
    cnp.ndarray[cnp.float64_t, ndim=1] q,
    float t
):
    return p + t * (q - p)


# @njit(cache=True)
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




# @njit(cache=True)
def eid_get_coefficient_i(event: EID) -> float:
    return event.t_i


# @njit(cache=True)
def eid_get_coefficient_j(event: EID) -> float:
    return event.t_j


cpdef EID from_coefficients(
    int i,
    int j,
    float t_p,
    float t_q,
    list P,
    list Q,
):

    """
    Create a new EID from coefficients. This shouldn't be
    used in a VE-Frechet algorithm, since this allows for
    edge-edge matchings.
    """

    cdef int n_p = len(P)
    cdef int n_q = len(Q)

    if not 0 <= i < n_p:
        raise ValueError(
            f'Cannot create event with index "{i}" on a curve with shape: {n_p}')

    if not 0 <= j < n_q:
        raise ValueError(
            f'Cannot create event with index "{j}" on a curve with shape: {n_q}'
        )

    cdef bint i_is_vert = False
    cdef Point p_i_res

    # Use this to avoid issues with floating point error
    if np.isclose(t_p, 0.0):
        p_i_res = P[i]
        i_is_vert = True
    elif np.isclose(t_p, 1.0):
        assert i + 1 < n_p
        i += 1
        t_p = 0.0
        p_i_res = P[i]
        i_is_vert = True
    else:
        assert i + 1 < n_p
        p_i_res = P[i].convex_comb(P[i+1], t_p)
        #Point(convex_comb(P[i], P[i + 1], t_p))

    cdef bint j_is_vert = False
    cdef Point p_j_res

    # Same as above, now for Q
    if np.isclose(t_q, 0.0):
        p_j_res = Q[j]
        j_is_vert = True
    elif np.isclose(t_q, 1.0):
        assert j + 1 < n_q
        j += 1
        t_q = 0.0
        p_j_res = Q[j]
        j_is_vert = True
    else:
        assert j + 1 < n_q
        p_j_res = Q[j].convex_comb(Q[j+1], t_q)
        #Point(convex_comb(Q[j], Q[j + 1], t_q))


    cdef float dist = p_i_res.compute_distance(p_j_res)

    return EID(i, i_is_vert, j, j_is_vert, p_i_res, p_j_res, t_p, t_q, dist)


def get_frechet_dist_from_morphing_list(morphing_list) -> float:
    res = 0.0

    for event in morphing_list:
        res = max(res, event.get_dist())

    return res



def _print_event_list(morphing: Morphing) -> None:
    for event in morphing.morphing_list:
        print(
            event.get_i(),
            event.get_i_is_vert(),
            event.get_j(),
            event.get_j_is_vert(),
            event.get_t_i(),
            event.get_t_j(),
            event.get_dist(),
        )


cpdef cnp.ndarray get_prefix_lens(list P):
    cdef int n = len(P)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] prefix_lens = np.empty(n)

    cdef double curr_len = 0.0

    for i in range(n - 1):
        prefix_lens[i] = curr_len
        curr_len += float(P[i].compute_distance(P[i + 1]))

    prefix_lens[n - 1] = curr_len

    return prefix_lens


# @njit(cache=True)
def eval_pl_func_on_dim(p: np.ndarray, q: np.ndarray, val: float, d: int) -> float:
    t = (val - p[d]) / (q[d] - p[d])
    return p * (1.0 - t) + q * t


# @njit(cache=True)
def eval_pl_func(p: np.ndarray, q: np.ndarray, val: float) -> float:
    #assert p.shape == q.shape
    assert p.shape[0] == q.shape[0] == 2
    return eval_pl_func_on_dim(p, q, val, 0)[1]


# @njit(cache=True)
def eval_inv_pl_func(p: np.ndarray, q: np.ndarray, val: float) -> float:
    #TODO bring back this assert?
    #assert p.shape == q.shape
    assert p.shape[0] == q.shape[0] == 2
    return eval_pl_func_on_dim(p, q, val, 1)[0]


cdef float coefficient_from_prefix_lens(
    float distance_along_curve,
    cnp.ndarray[cnp.float64_t, ndim=1] p_lens,
    int idx
):
    if idx == p_lens.shape[0] - 1:
        assert np.isclose(distance_along_curve, p_lens[idx])
        return 0.0
    elif np.isclose(distance_along_curve, p_lens[idx]):
        return 0.0

    assert p_lens[idx] <= distance_along_curve <= p_lens[idx + 1]

    cdef float edge_len = p_lens[idx + 1] - p_lens[idx]
    cdef float t = (distance_along_curve - p_lens[idx]) / edge_len

    return t


# @njit(cache=True)
cdef void assert_monotone_top(list prm):
    """
    Asserts monotonicity of the top of the PRM.
    """

    cdef int n = len(prm)
    if n < 2:
        return

    cdef tuple p = prm[-2]
    cdef tuple q = prm[-1]

    # Avoid raising exceptions on floating point jitters
    cdef double factor = 1.002

    if p[0] > factor * q[0] or p[1] > factor * q[1]:
        raise Exception(f"Monotonicity violated: {p}, {q}.")


cpdef list construct_new_prm(
    cnp.ndarray[cnp.float64_t, ndim=2] prm_1,
    cnp.ndarray[cnp.float64_t, ndim=2] prm_2
):

    cdef cnp.ndarray[cnp.float64_t, ndim=1] p_events
    cdef cnp.ndarray[cnp.float64_t, ndim=1] q_events_1
    cdef cnp.ndarray[cnp.float64_t, ndim=1] q_events_2
    cdef cnp.ndarray[cnp.float64_t, ndim=1] r_events

    q_events_1, r_events = prm_1
    p_events, q_events_2 = prm_2

    # print("PRM endings", q_events_1[-1], q_events_2[-1])
    # print(q_events_1.shape)
    # print(q_events_2.shape)
    assert np.allclose(q_events_1[-1], q_events_2[-1])

    cdef int idx_1 = 0
    cdef int idx_2 = 0

    cdef int len_1 = q_events_1.shape[0]
    cdef int len_2 = q_events_2.shape[0]

    cdef list new_prm = []

    # P = morphing_2.P
    # Q = morphing_2.Q = morphing_1.P
    # R = morphing_1.Q
    while idx_1 < len_1 - 1 or idx_2 < len_2 - 1:
        q_event_1 = q_events_1[idx_1]
        q_event_2 = q_events_2[idx_2]

        # print(idx_1, idx_2)
        # print("Two points: ", q_event_1, q_event_2)

        is_equal = np.isclose(q_event_1, q_event_2)

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
            #print(prm_2[:, idx_2 - 1].shape, prm_2[:, idx_2].shape)
            new_p = eval_inv_pl_func(prm_2[:, idx_2 - 1], prm_2[:, idx_2], q_event_1)
            # Enforcing monotonicity in the case of floating point error
            new_p = max(prm_2[:, idx_2 - 1][0], new_p)

            new_prm.append((new_p, r_events[idx_1]))
            idx_1 = min(idx_1 + 1, len_1 - 1)

        elif q_event_1 > q_event_2:
            # print("case6")
            new_r = eval_pl_func(prm_1[:, idx_1 - 1], prm_1[:, idx_1], q_event_2)
            # NOTE this line is needed because of extremely annoying floating point jitters
            new_r = max(prm_1[:, idx_1 - 1][1], new_r)
            new_prm.append((p_events[idx_2], new_r))
            idx_2 = min(idx_2 + 1, len_2 - 1)

        else:
            raise Exception("Should never get here")

        assert_monotone_top(new_prm)

    q_event_1 = q_events_1[idx_1]
    q_event_2 = q_events_2[idx_2]
    assert (
        idx_1 == len_1 - 1 and idx_2 == len_2 - 1 and np.isclose(q_event_1, q_event_2)
    )

    new_prm.append((p_events[idx_2], r_events[idx_1]))

    assert_monotone_top(new_prm)

    return new_prm


cpdef Morphing morphing_combine(
    Morphing morphing_1,
    Morphing morphing_2,
):
    # TODO this function only works on monotone morphings (I think). Use the
    # same helper function that Sariel used to fail when violations to the
    # monotonicity are found.
    # Code is based on:
    # https://github.com/sarielhp/FrechetDist.jl/blob/main/src/morphing.jl#L430

    print("Starting combine")

    cdef list P = morphing_2.P
    # Original curve equal to morphing_1.P
    assert morphing_1.P[-1].is_close(morphing_2.Q[-1])
    cdef list R = morphing_1.Q

    print("Getting PRMs")
    prm_1 = morphing_1.get_prm()
    prm_2 = morphing_2.get_prm()

    print("Constructing new PRMs")
    new_prm = construct_new_prm(prm_1, prm_2)
    print("About to create event sequence")
    return event_sequence_from_prm(new_prm, P, R)


cpdef Morphing event_sequence_from_prm(list prm, list P, list Q):
    cdef int i_p = 0
    cdef int i_q = 0

    #cdef list P_list = numpy_to_point_list(P)
    #cdef list Q_list = numpy_to_point_list(Q)

    cdef cnp.ndarray[cnp.float64_t, ndim=1] p_lens = get_prefix_lens(P)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] q_lens = get_prefix_lens(Q)

    cdef int p_num_pts = p_lens.shape[0]
    cdef int q_num_pts = q_lens.shape[0]

    cdef double max_dist = 0.0
    cdef list new_event_sequence = []

    cdef int i
    cdef float p_loc, q_loc, t_p, t_q
    cdef EID new_event

    for i in range(len(prm) - 1):
        # print(i)
        p_loc, q_loc = prm[i]

        while i_p < p_num_pts - 1 and p_loc >= p_lens[i_p + 1]:
            i_p += 1

        assert i_p == p_num_pts - 1 or p_lens[i_p] <= p_loc * 1.01

        while i_q < q_num_pts - 1 and q_loc >= q_lens[i_q + 1]:
            i_q += 1

        assert i_q == q_num_pts - 1 or q_lens[i_q] <= q_loc * 1.01

        t_p = coefficient_from_prefix_lens(p_loc, p_lens, i_p)
        t_q = coefficient_from_prefix_lens(q_loc, q_lens, i_q)
        # print(t_p, t_q)
        new_event = from_coefficients(i_p, i_q, t_p, t_q, P, Q)

        max_dist = max(max_dist, new_event.get_dist())
        new_event_sequence.append(new_event)
    # print("end event sequence")
    final_event = from_curve_indices(
        p_num_pts - 1, True, q_num_pts - 1, True, P, Q, None, None
    ).get_event()
    # print("actually done")
    max_dist = max(max_dist, final_event.get_dist())
    new_event_sequence.append(final_event)

    return Morphing(new_event_sequence, P, Q, max_dist)


def extract_offsets(
    P: np.ndarray, Q: np.ndarray, morphing: list[EID]
) -> tuple[np.ndarray, np.ndarray]:
    # I think this is radii without the vertex-vertex restriction
    P_offsets = np.zeros(P.shape[0], dtype=np.float64)
    Q_offsets = np.zeros(Q.shape[0], dtype=np.float64)

    for k in range(len(morphing)):
        event = morphing[k]
        P_offsets[event.i] = np.max(P_offsets[event.i], event.dist)  # type: ignore
        Q_offsets[event.j] = np.max(Q_offsets[event.j], event.dist)  # type: ignore

    return P_offsets, Q_offsets


# @njit(cache=True)
def simplify_polygon_radii(P: np.ndarray, r: np.ndarray) -> np.ndarray:
    assert P.shape[0] == r.shape[0]

    indices = [0]
    n = P.shape[0]

    curr = P[0]
    curr_r = r[0]
    for i in range(1, n):
        curr_r = min(curr_r, r[i])
        if np.linalg.norm(P[i] - curr) > curr_r:
            curr = P[i]
            if i < n - 1:
                curr_r = r[i + 1]
            indices.append(i)

    indices.append(n - 1)

    m = len(indices)
    d = P.shape[1]

    # Resulting curve will only have m indices
    P_simplified = np.zeros((m, d), dtype=np.float64)

    for i in range(m):
        P_simplified[i] = P[indices[i]]

    return P_simplified


# @njit(cache=True)
cpdef double frechet_dist_upper_bound(
    cnp.ndarray P,
    cnp.ndarray Q,
):
    """
    Returns a rough upper bound on the Frechet distance between the two
    curves. This upper bound is on the continuous distance. No guarentee
    on how bad the approximation is. This is used as a starting point for
    real approximation of the Frechet distance, and should not be used
    otherwise.
    """

    w_a = frechet_width_approx(P)
    w_b = frechet_width_approx(Q)

    if P.shape[0] <= 2 or Q.shape[0] <= 2:
        return w_a + w_b

    w = max(np.linalg.norm(P[0] - Q[0]), np.linalg.norm(P[-1] - Q[-1]))

    return w_a + w_b + w


# @njit(cache=True)
def frechet_width_approx(
    P: np.ndarray, idx_range: tuple[int, int] | None = None
) -> float:
    # TODO write some test code for this bc the indexing might be off
    """
    2-approximation to the Frechet distance between
    P[first(rng)]-P[last(rng)] and he polygon
    P[rng]
    Here, rng is a range i:j
    """

    if idx_range is None:
        start, end = 0, P.shape[0]
    else:
        start, end = idx_range

    if end - start <= 2:
        return 0.0

    start_point = P[start]
    end_point = P[end - 1]

    leash = 0.0
    t = 0.0
    curr = start_point

    # TODO double check w/ Sariel because this seems like a weird min condition
    for i in range(start + 1, end - 1):
        p = P[i]
        _, new_t, q = line_point_distance(start_point, end_point, p)

        if new_t > t:
            t = new_t
            curr = q

        leash = max(leash, float(np.linalg.norm(curr - p)))

    return leash

cdef class NewCurves:
    def __cinit__(self, P, Q):
        self.P = P
        self.Q = Q

    cpdef cnp.ndarray get_P_numpy(self):
        new_P_final = np.empty((len(self.P), len(self.P[0].get_coords())))

        for k in range(len(self.P)):
            new_P_final[k] = self.P[k].get_coords()

        return new_P_final

    cpdef cnp.ndarray get_Q_numpy(self):
        new_Q_final = np.empty((len(self.Q), len(self.Q[0].get_coords())))

        for k in range(len(self.Q)):
            new_Q_final[k] = self.Q[k].get_coords()

        return new_Q_final

    cpdef list get_P(self):
        return self.P

    cpdef list get_Q(self):
        return self.Q

cpdef NewCurves add_points_to_make_monotone(Morphing morphing):
    # TODO add intermediate vertices here
    # Doing the same here:
    # https://github.com/sarielhp/FrechetDist.jl/blob/main/src/frechet.jl#L626

    cdef list P = morphing.P
    cdef list Q = morphing.Q
    cdef list morphing_list = morphing.morphing_list
    # print(len(morphing_list))
    # First, add points to P
    cdef list new_P = []
    cdef list events
    cdef int k = 0
    cdef int loc
    cdef bint monotone

    while k < len(morphing_list):
        # Vertex-vertex event, can skip
        if morphing_list[k].get_i_is_vert():
            new_P.append(P[morphing_list[k].get_i()])
            old_k = k
            while (
                k < len(morphing_list)
                and morphing_list[old_k].get_i() == morphing_list[k].get_i()
                and morphing_list[k].get_i_is_vert()
            ):
                k += 1
            continue

        loc = morphing_list[k].get_i()
        events = []

        # [old_k,k) is the indices of points that are on the same segment
        # So increase new_k to get the max window where this is the case
        while (
            k < len(morphing_list)
            and not morphing_list[k].get_i_is_vert()
            and morphing_list[k].get_i() == loc
        ):
            events.append(morphing_list[k])
            k += 1

        # Next, check if the offsets are monotone as-given
        monotone = True
        for j in range(len(events) - 1):
            monotone = monotone and (events[j].get_t_i() <= events[j + 1].get_t_i())

        # TODO double check this is the right thing to do
        if monotone:
            continue

        events = sorted(events, key=eid_get_coefficient_i)

        if not monotone:
            # NOTE Use i because we know we're not at the vertex from case checked above
            new_P.append((P[events[0].get_i()].get_avg(events[0].get_p_i())))

        for j in range(len(events)):
            new_P.append(events[j].get_p_i())

            if not monotone and j < len(events) - 1:
                # print("Adding average: ", events[j].p_i, events[j + 1].p_i)
                new_P.append(events[j].get_p_i().get_avg(events[j + 1].get_p_i()))

        if not monotone and events[-1].get_i() + 1 < len(P):
            new_P.append(P[events[-1].get_i() + 1].get_avg(events[-1].get_p_i()))

    # # Next, add points to Q, same as above but hard to share logic
    cdef list new_Q = []
    k = 0
    while k < len(morphing_list):
        if morphing_list[k].get_j_is_vert():
            new_Q.append(Q[morphing_list[k].get_j()])
            old_k = k
            while (
                k < len(morphing_list)
                and morphing_list[old_k].get_j() == morphing_list[k].get_j()
                and morphing_list[k].get_j_is_vert()
            ):
                k += 1
            continue

        loc = morphing_list[k].get_j()
        events = []

        # [old_k,k) is the indices of points that are on the same segment
        # So increase new_k to get the max window where this is the case
        while (
            k < len(morphing_list)
            and not morphing_list[k].get_j_is_vert()
            and morphing_list[k].get_j() == loc
        ):
            events.append(morphing_list[k])
            k += 1

        # Next, check if the offsets are monotone as-given
        monotone = True
        for j in range(len(events) - 1):
            monotone = monotone and (events[j].get_t_j() <= events[j + 1].get_t_j())

        if monotone:
            continue

        events = sorted(events, key=eid_get_coefficient_j)

        if not monotone:
            # NOTE Use j because we know we're not at the vertex from case checked above
            new_Q.append(Q[events[0].get_j()].get_avg(events[0].get_p_j()))

        for j in range(len(events)):
            new_Q.append(events[j].get_p_j())

            if not monotone and j < len(events) - 1:
                # print("Adding average: ", events[j].p_i, events[j + 1].p_i)
                new_Q.append(events[j].get_p_j().get_avg(events[j + 1].get_p_j()))

        if not monotone and events[-1].get_j() + 1 < len(Q):
            new_Q.append(Q[events[-1].get_j() + 1].get_avg(events[-1].get_p_j()))

    # Finally, assemble into output arrays
    # new_P_final = np.empty((len(new_P), len(new_P[0].get_coords())))
    # new_Q_final = np.empty((len(new_Q), len(new_Q[0].get_coords())))

    # for k in range(len(new_P)):
    #     new_P_final[k] = new_P[k].get_coords()

    # for k in range(len(new_Q)):
    #     new_Q_final[k] = new_Q[k].get_coords()

    return NewCurves(new_P, new_Q)
    #return new_P_final, new_Q_final


# @njit
def line_line_distance(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> float:
    """
    Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
    Return the closest points on each segment and their distance
    From: https://stackoverflow.com/a/18994296/2923069
    """

    v_1 = a0 - b0
    v_2 = a1 - a0
    v_3 = b0 - b1

    # D(s,t)
    # = <v_1 + s*v_2 + t *v_3, v_1 + s*v_2 + t *v_3>
    # = ||v_1||^2  +  2 * s * <v_1, v_2>  +  2 * t *<v_1,v_3>
    #    + s^2 * ||v_2||^2 + 2*s*t*<v_2,v_3>  + t^2 ||v_3||^2
    # =
    # Need to solve the linear system:

    # 0 = D'_s(s,t) = 2*<v_1,v_2> + 2*s*||V_2||^2  +  2*t<v_2,v_3>
    # 0 = D'_t(s,t) = 2*<v_1,v_3> + 2*s*<v_2,v_3> + 2*t * ||v_3||^2

    # or equivalently:

    # -<v_1,v_2> = s * ||V_2||^2  +  t * <v_2,v_3>
    # -<v_1,v_3> = s * <v_2,v_3>  +  t * ||v_3||^2

    c = np.array([-np.dot(v_1, v_2), -np.dot(v_1, v_3)])

    m = np.array(
        [[np.dot(v_2, v_2), np.dot(v_2, v_3)], [np.dot(v_2, v_3), np.dot(v_3, v_3)]]
    )

    if np.linalg.matrix_rank(m) == 1:
        # The minimum distance is realized by one of the endpoints.
        return min(
            line_point_distance(a0, a1, b0)[0],
            line_point_distance(a0, a1, b1)[0],
            line_point_distance(b0, b1, a0)[0],
            line_point_distance(b0, b1, a1)[0],
        )

        # Solve the system...
    b = np.linalg.solve(m, c)
    # println( "b.size: ", size( b ) );
    s = b[0]
    t = b[1]

    # Snap solution if needed to the [0,1.0] interval....
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)

    d = float(np.linalg.norm(convex_comb(a0, a1, s) - convex_comb(b0, b1, t)))

    # d::Float64 =  Dist( convex_comb( a0, a1, s ),
    #    convex_comb( b0, b1, t ) )

    d = min(
        d,
        line_point_distance(a0, a1, b0)[0],
        line_point_distance(a0, a1, b1)[0],
        line_point_distance(b0, b1, a0)[0],
        line_point_distance(b0, b1, a1)[0],
    )

    return d
