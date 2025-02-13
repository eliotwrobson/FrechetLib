cimport numpy as cnp
from .frechet_utils cimport Morphing


cpdef Morphing retractable_ve_frechet(
    cnp.ndarray[cnp.float64_t, ndim=2] P,
    cnp.ndarray[cnp.float64_t, ndim=2] Q,
    cnp.ndarray[cnp.float64_t, ndim=1] P_offs,
    cnp.ndarray[cnp.float64_t, ndim=1] Q_offs,
    bint summed,
)

cpdef Morphing retractable_ve_frechet_internal(
    list P,
    list Q,
    cnp.ndarray[cnp.float64_t, ndim=1] P_offs,
    cnp.ndarray[cnp.float64_t, ndim=1] Q_offs,
    bint summed,
)
