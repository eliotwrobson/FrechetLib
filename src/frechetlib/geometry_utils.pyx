cimport numpy as np
from libcpp.vector cimport vector as cvector
cimport cython

cdef bint double_equals(double a, double b, double epsilon = 1e-10):
  return abs(a - b) < epsilon

@cython.final
cdef class Point:
    def __cinit__(self, double[:] coords):
        for entry in coords:
            self.coords.push_back(entry)

    def __cinit__(self, cvector[double] coords):
        for entry in coords:
            self.coords.push_back(entry)

    def get_coords(self):
        return list(self.coords)

    cpdef inline Point convex_comb(self, Point q, double t):
        cdef cvector[double] new_coords

        cdef int i
        for i in range(len(self.coords)):
            new_coords.push_back(self.coords[i] + t * (q.coords[i] - self.coords[i]))

        return Point(new_coords)

    cpdef inline Point point_difference(self, Point q):
        cdef cvector[double] new_coords

        cdef int i
        for i in range(len(self.coords)):
            new_coords.push_back(self.coords[i] - q.coords[i])

        return Point(new_coords)

    cpdef inline double get_norm(self):
        cdef double norm = 0.0
        cdef int i
        for i in range(len(self.coords)):
            norm += self.coords[i] ** 2

        return norm ** 0.5

    cpdef inline double compute_distance(self, Point q):
        cdef double dist = 0.0
        cdef int i

        for i in range(len(self.coords)):
            dist += (self.coords[i] - q.coords[i]) ** 2

        return dist ** 0.5

    cpdef inline double dot_product(self, Point q):
        cdef double dot = 0.0
        cdef int i
        for i in range(len(self.coords)):
            dot += self.coords[i] * q.coords[i]

        return dot

    cpdef inline bint is_close(self, Point q):
        cdef bint res = True
        cdef int i

        for i in range(len(self.coords)):
            res &= double_equals(self.coords[i], q.coords[i])

        return res

    cpdef inline Point get_avg(self, Point q):
        cdef cvector[double] new_coords

        cdef int i
        for i in range(len(self.coords)):
            new_coords.push_back((self.coords[i] + q.coords[i])/2.0)

        return Point(new_coords)

@cython.final
cdef class LinePointDistance:
    def __cinit__(self, p1: np.ndarray, p2: np.ndarray, q: np.ndarray):
        self.compute(Point(p1), Point(p2), Point(q))

    cdef inline void compute(self, Point p1, Point p2, Point q):
        #cdef Point point_p1 = Point(p1)
        #cdef Point point_p2 = Point(p2)

        cdef Point q_diff = q.point_difference(p1)
        cdef Point p_diff = p2.point_difference(p1)

        cdef double l2 = p1.compute_distance(p2) ** 2  # i.e. |p2-p1|^2
        if double_equals(l2, 0.0):  # p1 == p2 case
            self.distance = q.compute_distance(p1)
            self.t = 0.0
            self.p = p1
            return

        # Consider the line extending the segment, parameterized as v + t (p2 - p1).
        # We find projection of point q onto the line.
        # It falls where t = [(q-p1) . (p2-p1)] / |p2-p1|^2
        # We clamp t from [0,1] to handle points outside the segment vw.
        t = q_diff.dot_product(p_diff) / l2

        if t <= 0.0:
            self.distance = q.compute_distance(p1)
            self.t = 0.0
            self.p = p1
            return
        elif t >= 1.0:
            self.distance = q.compute_distance(p2)
            self.t = 1.0
            self.p = p2
            return
        else:
            self.p = p1.convex_comb(p2, t)
            self.t = t
            self.distance = q.compute_distance(self.p)

    cpdef double get_distance(self):
        return self.distance

    cpdef double get_t(self):
        return self.t

    cpdef Point get_p(self):
        return self.p

cdef class EID:
    # i: int
    # i_is_vert: bool
    # j: int
    # j_is_vert: bool

    # # Computed distance between the points
    # dist: float

    # # Points on edges if not a vertex. Can be adjusted.
    # p_i: np.ndarray
    # p_j: np.ndarray

    # # Parameters for the points on each curve.
    # t_i: float
    # t_j: float

    def __cinit__(
        self,
        i: int,
        i_is_vert: bool,
        j: int,
        j_is_vert: bool,
        p_i: Point,
        p_j: Point,
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

        assert 0.0 <= t_i <= 1.0
        assert 0.0 <= t_j <= 1.0

    # TODO get rid of these getter functions later
    def get_dist(self):
        return self.dist

    def get_i(self):
        return self.i

    def get_j(self):
        return self.j

    def get_i_is_vert(self):
        return self.i_is_vert

    def get_j_is_vert(self):
        return self.j_is_vert

    def get_t_i(self):
        return self.t_i

    def get_t_j(self):
        return self.t_j

    def get_p_i(self):
        return self.p_i

    def get_p_j(self):
        return self.p_j

    cpdef EID copy(self):
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

    cpdef float reassign_parameter_i(
        self,
        float new_t,
        np.ndarray[np.float64_t, ndim=2] P
    ):
        """
        Reassign the point and parameter from the curve P.
        Returns the error incurred by the reassignment.
        """
        assert 0.0 <= new_t <= 1.0
        cdef double old_t = self.t_i

        if double_equals(0.0, new_t):
            self.t_i = 0.0
            self.p_i = Point(P[self.i])
        elif double_equals(1.0, new_t):
            self.t_i = 1.0
            self.p_i = Point(P[self.i + 1])
            # TODO maybe change the number based on index?
            # I don't think the convention matters.
        else:
            # Case where 0.0 < new_t < 1.0
            self.t_i = new_t
            self.p_i = Point(P[self.i]).convex_comb(Point(P[self.i + 1]), self.t_i)

        self.dist = self.p_i.compute_distance(self.p_j)
        return abs(old_t - new_t) * Point(P[self.i]).compute_distance(Point(P[self.i + 1]))

    cpdef float reassign_parameter_j(
        self,
        float new_t,
        np.ndarray[np.float64_t, ndim=2] Q
    ):
        """
        Reassign the point and parameter from the curve Q.
        Returns the error incurred by the reassignment.
        """
        assert 0.0 <= new_t <= 1.0
        cdef double old_t = self.t_j

        if double_equals(0.0, new_t):
            self.t_j = 0.0
            self.p_j = Point(Q[self.j])
        elif double_equals(1.0, new_t):
            self.t_j = 1.0
            self.p_j = Point(Q[self.j + 1])
            # TODO maybe change the number based on index?
            # I don't think the convention matters.
        else:
            # Case where 0.0 < new_t < 1.0
            self.t_j = new_t
            self.p_j = Point(Q[self.j]).convex_comb(Point(Q[self.j + 1]), self.t_j)

        self.dist = self.p_i.compute_distance(self.p_j)
        return abs(old_t - new_t) * Point(Q[self.i]).compute_distance(Point(Q[self.i + 1]))

    cpdef flip(self):
        self.i, self.j = self.j, self.i
        self.i_is_vert, self.j_is_vert = self.j_is_vert, self.i_is_vert
        self.p_i, self.p_j = self.p_j, self.p_i
        self.t_i, self.t_j = self.t_j, self.t_i

    def __lt__(self, other: Self) -> bool:
        # This function is mainly used to schedule events for heap insertion.
        # TODO when this project gets refactored, get rid of this function and
        # just manually compute the key used in the heap.
        return self.dist < other.get_dist()

    def __hash__(self) -> int:
        return hash(
            (
                self.i,
                self.i_is_vert,
                self.j,
                self.j_is_vert,
                self.t_i,
                self.t_j,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EID):
            return False

        return (
            (self.i == other.get_i())
            and (self.j == other.get_j())
            and (self.i_is_vert == other.get_i_is_vert())
            and (self.j_is_vert == other.get_j_is_vert())
            and bool(self.p_i.is_close(other.get_p_i()))
            and bool(self.p_j.is_close(other.get_p_j()))
            and bool(double_equals(self.t_i, other.get_t_i()))
            and bool(double_equals(self.t_j, other.get_t_j()))
        )

@cython.final
cdef class EIDFromCurveIndices:
    def __cinit__(
        self,
        heap_key: double,
        event: EID,
    ) -> None:
        self.heap_key = heap_key
        self.event = event

    cpdef double get_heap_key(self):
        return self.heap_key

    cpdef EID get_event(self):
        return self.event

cpdef EIDFromCurveIndices from_curve_indices(
    int i,
    bint i_is_vert: bool,
    int j,
    bint j_is_vert: bool,
    np.ndarray P,
    np.ndarray Q,
    np.ndarray P_offs,
    np.ndarray Q_offs,
):

    # These values will get overwritten later
    # TODO I think some of the logic below can be refactored to reduce
    # the number of cases
    dist = 0.0
    heap_key = 0.0
    t_i = 0.0
    t_j = 0.0
    p_i = Point(P[i])
    p_j = Point(Q[j])

    if not 0 <= i < P.shape[0]:
        raise ValueError(
            f'Cannot create event with index "{i}" on a curve with shape:'
            f"{P.shape[0]}, {P.shape[1]}."
        )

    if not 0 <= j < Q.shape[0]:
        raise ValueError(
            f'Cannot create event with index "{j}" on a curve with shape:'
            f"{Q.shape[0]}, {Q.shape[1]}."
        )

    use_offsets = P_offs is not None and Q_offs is not None

    if use_offsets:
        assert P.shape[0] == P_offs.shape[0]  # type: ignore[union-attr]
        # print("shapes", P.shape, P_offs.shape, Q.shape, Q_offs.shape)
        assert Q.shape[0] == Q_offs.shape[0]  # type: ignore[union-attr]

    if i_is_vert and j_is_vert:
        dist = Point(P[i]).compute_distance(Point(Q[j]))

        if use_offsets:
            heap_key = dist - P_offs[i] - Q_offs[j]  # type: ignore[index]
        else:
            heap_key = dist

    elif i_is_vert:
        if j == Q.shape[0] - 1:
            dist = Point(P[i]).compute_distance(Point(Q[j]))

            if use_offsets:
                heap_key = dist - P_offs[i] - Q_offs[j]  # type: ignore[index]
            else:
                heap_key = dist
        else:
            dist_obj = LinePointDistance(Q[j], Q[j + 1], P[i])
            dist = dist_obj.get_distance()
            t_j = dist_obj.get_t()
            p_j = dist_obj.get_p()

            #dist, t_j, p_j = LinePointDistance(Q[j], Q[j + 1], P[i])

            #line_point_distance(Q[j], Q[j + 1], P[i])

            if use_offsets:
                heap_key = dist - P_offs[i] - max(Q_offs[j], Q_offs[j + 1])  # type: ignore[index]
            else:
                heap_key = dist

    elif j_is_vert:
        if i == P.shape[0] - 1:
            dist = Point(P[i]).compute_distance(Point(Q[j]))
            #float(np.linalg.norm(P[i] - Q[j]))

            if use_offsets:
                heap_key = dist - P_offs[i] - Q_offs[j]  # type: ignore[index]
            else:
                heap_key = dist
        else:
            dist_obj = LinePointDistance(P[i], P[i + 1], Q[j])
            dist = dist_obj.get_distance()
            t_i = dist_obj.get_t()
            p_i = dist_obj.get_p()

            #dist, t_i, p_i = LinePointDistance(P[i], P[i + 1], Q[j])
            #line_point_distance(P[i], P[i + 1], Q[j])

            if use_offsets:
                heap_key = dist - max(P_offs[i], P_offs[i + 1]) - Q_offs[j]  # type: ignore[index]
            else:
                heap_key = dist
    else:
        raise Exception

    assert 0.0 <= t_i <= 1.0
    assert 0.0 <= t_j <= 1.0

    # TODO figure out how to use offsets as the key.
    return EIDFromCurveIndices(heap_key, EID(i, i_is_vert, j, j_is_vert, p_i, p_j, t_i, t_j, dist))


#### Start of functions mainly used for testing ####

def convex_comb(p: np.ndarray, q: np.ndarray, t: float):
    return Point(p).convex_comb(Point(q), t).get_coords()
    #return p + t * (q - p)
