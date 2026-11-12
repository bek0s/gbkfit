"""
The observations: how the model of galaxies gives the data of a dataset
(see Observation and observation_parser). An observation on the sky
(ObservationSky) sees galaxies as an observable (see
gbkfit.observation.observables, whose observables it exports too),
along a line of sight, through a foreground (e.g. a gravitational lens;
see gbkfit.observation.lensing) and an instrument; an observation of
profiles (ObservationProfile) gives values of functions of a galaxy at
radii. Observations are evaluated in groups (see ObservationGroup and
GroupPlan), compared with their data under their likelihoods (see
Likelihood and Comparison), and draw mock data (see draw_mocks).
"""

from . import (
    comparisons, foreground, group, group_plan, lensing, likelihoods,
    line_of_sight, mocks, noise, observables, observation, profile, sky,
    weights)
from .comparisons import *
from .foreground import *
from .group import *
from .group_plan import *
from .lensing import *
from .likelihoods import *
from .line_of_sight import *
from .mocks import *
from .noise import *
from .observables import *
from .observation import *
from .profile import *
from .sky import *
from .weights import *


__all__ = []
__all__ += comparisons.__all__
__all__ += foreground.__all__
__all__ += group.__all__
__all__ += group_plan.__all__
__all__ += lensing.__all__
__all__ += likelihoods.__all__
__all__ += line_of_sight.__all__
__all__ += mocks.__all__
__all__ += noise.__all__
__all__ += observables.__all__
__all__ += observation.__all__
__all__ += profile.__all__
__all__ += sky.__all__
__all__ += weights.__all__


def _register_observations() -> None:
    from gbkfit.observation.observation import (
        observation_parser as abstract_parser)
    abstract_parser.register([
        ObservationProfile,
        ObservationSky
    ])


_register_observations()
