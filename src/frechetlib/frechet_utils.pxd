cimport numpy as cnp
from libcpp.vector cimport vector as cvector

ctypedef cnp.float64_t FLOAT_t

cdef class Morphing:
    #TODO figure out how to get this to work
    cdef object morphing_list
    cdef cnp.ndarray P
    cdef cnp.ndarray Q
    cdef float dist

    cpdef flip(self)

    cpdef Morphing copy(self)

    cpdef bint is_monotone(self)

    cpdef FLOAT_t make_monotone(self)
