cimport numpy as np

cdef class EID:
    cdef int i
    cdef bint i_is_vert
    cdef int j
    cdef bint j_is_vert

    # Computed distance between the points
    cdef float dist

    # Points on edges if not a vertex. Can be adjusted.
    cdef np.ndarray p_i
    cdef np.ndarray p_j

    # Parameters for the points on each curve.
    cdef float t_i
    cdef float t_j
