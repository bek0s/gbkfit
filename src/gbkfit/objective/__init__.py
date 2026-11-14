"""
The objective of a fit: the comparison of the models of a group of
observations with their data under their likelihoods (see Objective,
and Likelihood and Likelihoods), which solves the elements of
parameters that the models are linear in (flux scales, amplitudes of
light and weights of orbits) at each evaluation, and its evaluations and
merits (see ObjectiveEvaluation and Merits). The comparison of each
observation is of the kind of its likelihood (see Comparison and
comparison_of), and whitens the residuals of correlated noise (see
Whitening).
"""

from . import _comparisons, _whitening, likelihoods, objective
from ._comparisons import *
from ._whitening import *
from .likelihoods import *
from .objective import *


__all__ = []
__all__ += _comparisons.__all__
__all__ += _whitening.__all__
__all__ += likelihoods.__all__
__all__ += objective.__all__
