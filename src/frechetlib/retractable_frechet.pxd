cimport numpy as cnp
from .frechet_utils cimport Morphing


cpdef Morphing retractable_ve_frechet(
    cnp.ndarray P,
    cnp.ndarray Q,
    cnp.ndarray P_offs,
    cnp.ndarray Q_offs,
    bint summed,
)
