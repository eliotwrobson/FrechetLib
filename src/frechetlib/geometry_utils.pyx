cimport numpy as np
from libcpp.vector cimport vector as cvector
cimport cython

cdef  bint double_equals(double a, double b, double epsilon = 1e-10):
  return abs(a - b) < epsilon

@cython.final
cdef class Point:
    #cdef cvector[float] coords

    def __cinit__(self, double[:] coords):
        for entry in coords:
            self.coords.push_back(entry)

    def __cinit__(self, cvector[double] coords):
        for entry in coords:
            self.coords.push_back(entry)

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

    def get_coords(self):
        return list(self.coords)

    cpdef inline double get_norm(self):
        cdef double norm = 0.0
        cdef int i
        for i in range(len(self.coords)):
            norm += self.coords[i] ** 2

        return norm ** 0.5

    cpdef inline double dot_product(self, Point q):
        cdef double dot = 0.0
        cdef int i
        for i in range(len(self.coords)):
            dot += self.coords[i] * q.coords[i]

        return dot

@cython.final
cdef class LinePointDistance:

    def __cinit__(self, p1: np.ndarray, p2: np.ndarray, q: np.ndarray):
        self.compute(Point(p1), Point(p2), Point(q))

    cdef inline void compute(self, Point p1, Point p2, Point q):
        #cdef Point point_p1 = Point(p1)
        #cdef Point point_p2 = Point(p2)

        cdef Point q_diff = q.point_difference(p1)
        cdef Point p_diff = p2.point_difference(p1)

        cdef double l2 = p1.point_difference(p2).get_norm() ** 2  # i.e. |p2-p1|^2
        if double_equals(l2, 0.0):  # p1 == p2 case
            self.distance = q.point_difference(p1).get_norm()
            self.t = 0.0
            self.p = p1
            return

        # Consider the line extending the segment, parameterized as v + t (p2 - p1).
        # We find projection of point q onto the line.
        # It falls where t = [(q-p1) . (p2-p1)] / |p2-p1|^2
        # We clamp t from [0,1] to handle points outside the segment vw.
        t = q_diff.dot_product(p_diff) / l2

        if t <= 0.0:
            self.distance = q.point_difference(p1).get_norm()
            self.t = 0.0
            self.p = p1
            return
        elif t >= 1.0:
            self.distance = q.point_difference(p2).get_norm()
            self.t = 1.0
            self.p = p2
            return
        else:
            self.p = p1.convex_comb(p2, t)
            self.t = t
            self.distance = q.point_difference(self.p).get_norm()

    cpdef double get_distance(self):
        return self.distance

    cpdef double get_t(self):
        return self.t

    cpdef Point get_p(self):
        return self.p



#### Start of functions mainly used for testing ####

def convex_comb(p: np.ndarray, q: np.ndarray, t: float):
    return Point(p).convex_comb(Point(q), t).get_coords()
    #return p + t * (q - p)
