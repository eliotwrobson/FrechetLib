cimport numpy as cnp

ctypedef cnp.float64_t FLOAT_t

cdef class EID:
    cdef int i
    cdef bint i_is_vert
    cdef int j
    cdef bint j_is_vert

    # Computed distance between the points
    cdef float dist

    # Points on edges if not a vertex. Can be adjusted.
    cdef cnp.ndarray p_i
    cdef cnp.ndarray p_j

    # Parameters for the points on each curve.
    cdef float t_i
    cdef float t_j

    cpdef float reassign_parameter_i(
        self,
        float new_t,
        cnp.ndarray[FLOAT_t, ndim=2] P
    )

    cpdef float reassign_parameter_j(
        self,
        float new_t,
        cnp.ndarray[FLOAT_t, ndim=2] Q
    )


cdef class Morphing:
    #TODO figure out how to get this to work
    cdef object morphing_list
    cdef cnp.ndarray P
    cdef cnp.ndarray Q
    cdef float dist
