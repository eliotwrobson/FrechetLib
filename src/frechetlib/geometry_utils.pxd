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

    cpdef inline bint is_close(self, Point q)

    cpdef inline Point get_avg(self, Point q)

@cython.final
cdef class LinePointDistance:
    cdef double distance
    cdef double t
    cdef Point p

    cdef inline void compute(self, Point p1, Point p2, Point q)

    cpdef double get_distance(self)

    cpdef double get_t(self)

    cpdef Point get_p(self)

cdef class EID:
    cdef int i
    cdef bint i_is_vert
    cdef int j
    cdef bint j_is_vert

    # Computed distance between the points
    cdef float dist

    # Points on edges if not a vertex. Can be adjusted.
    cdef Point p_i
    cdef Point p_j

    # Parameters for the points on each curve.
    cdef float t_i
    cdef float t_j

    cpdef float reassign_parameter_i(
        self,
        float new_t,
        np.ndarray[np.float64_t, ndim=2] P
    )

    cpdef float reassign_parameter_j(
        self,
        float new_t,
        np.ndarray[np.float64_t, ndim=2] Q
    )

    cpdef flip(self)

    cpdef EID copy(self)

@cython.final
cdef class EIDFromCurveIndices:
    cdef double heap_key
    cdef EID event

    cpdef double get_heap_key(self)

    cpdef EID get_event(self)

cpdef EIDFromCurveIndices from_curve_indices(
    int i,
    bint i_is_vert,
    int j,
    bint j_is_vert,
    np.ndarray P,
    np.ndarray Q,
    np.ndarray P_offs,
    np.ndarray Q_offs,
)
