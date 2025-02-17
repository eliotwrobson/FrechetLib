cimport numpy as cnp
from .frechet_utils cimport Morphing
cimport cython


cpdef Morphing frechet_mono_via_refinement(
    cnp.ndarray P, cnp.ndarray Q, float approx
)

@cython.final
cdef class FrechetApproxResult:
    cdef double ratio
    cdef Morphing morphing

    cpdef double get_ratio(self)

    cpdef Morphing get_morphing(self)

cpdef FrechetApproxResult frechet_c_approx(
    cnp.ndarray P, cnp.ndarray Q, float approx_ratio
)
