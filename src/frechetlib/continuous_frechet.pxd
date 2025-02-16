cimport numpy as cnp
from .frechet_utils cimport Morphing


cpdef Morphing frechet_mono_via_refinement(
    cnp.ndarray P, cnp.ndarray Q, float approx
)
