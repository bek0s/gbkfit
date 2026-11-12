"""
The objective of a fit: the comparison of the models of a group of
observations with their data (see Objective), which solves the elements
of parameters that the models are linear in (flux scales, amplitudes of
light and weights of orbits) at each evaluation, and its evaluations and
merits (see ObjectiveEvaluation and Merits).
"""

from . import objective
from .objective import *


__all__ = []
__all__ += objective.__all__
