"""
The objective of a fit: the comparison of the models of a group of
observations with their data (see Objective), which solves the elements
of parameters that the models are linear in (flux scales, amplitudes of
light and weights of orbits) at each evaluation, and its evaluations and
merits (see ObjectiveEvaluation and Merits). The comparison of each
observation is of the kind of its likelihood (see Comparison and
comparison_of), and whitens the residuals of correlated noise (see
Whitening).
"""

from . import _comparisons, _whitening, objective
from ._comparisons import *
from ._whitening import *
from .objective import *


__all__ = []
__all__ += _comparisons.__all__
__all__ += _whitening.__all__
__all__ += objective.__all__
