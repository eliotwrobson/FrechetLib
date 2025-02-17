cimport numpy as cnp
from libcpp.vector cimport vector as cvector
from .geometry_utils cimport EID

cdef class Curve:
    cdef list point_list

cdef class Morphing:
    #TODO figure out how to get this to work
    cdef list morphing_list
    cdef list P
    cdef list Q
    cdef float dist

    cpdef flip(self)

    cpdef Morphing copy(self)

    cpdef bint is_monotone(self)

    cpdef float make_monotone(self)

cdef class NewCurves:
    cdef list P
    cdef list Q

    cpdef cnp.ndarray get_P_numpy(self)

    cpdef cnp.ndarray get_Q_numpy(self)

    cpdef list get_P(self)

    cpdef list get_Q(self)

cpdef NewCurves add_points_to_make_monotone(Morphing morphing)

cpdef EID from_coefficients(
    int i,
    int j,
    float t_p,
    float t_q,
    list P,
    list Q,
)

cpdef cnp.ndarray get_prefix_lens(list P)

cpdef Morphing event_sequence_from_prm(list prm, list P, list Q)

cpdef Morphing morphing_combine(
    Morphing morphing_1,
    Morphing morphing_2,
)

cpdef double frechet_dist_upper_bound(
    cnp.ndarray P,
    cnp.ndarray Q,
)
