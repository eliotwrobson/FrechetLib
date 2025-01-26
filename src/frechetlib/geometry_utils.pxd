from libcpp.vector cimport vector as cvector
cimport numpy as np
cimport cython

@cython.final
cdef class Point:
    cdef cvector[double] coords

    #def __cinit__(self, cvector[float] coords)

    cpdef inline Point convex_comb(self, Point q, double t)

    cpdef inline Point point_difference(self, Point q)

    cpdef inline double get_norm(self)

    cpdef inline double dot_product(self, Point q)

@cython.final
cdef class LinePointDistance:
    cdef double distance
    cdef double t
    cdef Point p

    cdef inline void compute(self, Point p1, Point p2, Point q)

    cpdef double get_distance(self)

    cpdef double get_t(self)

    cpdef Point get_p(self)
