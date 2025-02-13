cimport numpy as cnp
from libcpp.vector cimport vector as cvector

cdef class Curve:
    cdef list point_list

cdef class Morphing:
    #TODO figure out how to get this to work
    cdef list morphing_list
    cdef cnp.ndarray P
    cdef cnp.ndarray Q
    cdef float dist

    cpdef flip(self)

    cpdef Morphing copy(self)

    cpdef bint is_monotone(self)

    cpdef float make_monotone(self)

cdef class NewCurves:
    cdef cnp.ndarray P
    cdef cnp.ndarray Q

    cpdef cnp.ndarray[cnp.float64_t, ndim=2] get_P(self)

    cpdef cnp.ndarray[cnp.float64_t, ndim=2] get_Q(self)

cpdef NewCurves add_points_to_make_monotone(Morphing morphing)
