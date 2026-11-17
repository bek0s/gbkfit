"""
The observations: how the model of galaxies gives the data of a dataset
(see Observation and observation_parser). An observation on the sky
(see ObservationSky, of one kind for each kind of data: see
gbkfit.observation.sky, whose kinds it exports too) sees galaxies along
the line of sight (on a depth axis if what it sees needs one, see
Depth), through the gravitational lenses in front of them (see
gbkfit.lensing) and an instrument; an observation of
profiles (ObservationProfile) gives values of functions of a galaxy at
radii. Observations are evaluated in groups (see ObservationGroup and
GroupPlan), those with data compared with them under the likelihoods
that their setup gives them (see gbkfit.objective), and draw mock data
(see draw_mocks).
"""

from . import (
    depth, group, group_plan, mocks, noise, observation, plan, profile, sky,
    weights)
from .depth import *
from .group import *
from .group_plan import *
from .mocks import *
from .noise import *
from .observation import *
from .plan import *
from .profile import *
from .sky import *
from .weights import *


__all__ = []
__all__ += depth.__all__
__all__ += group.__all__
__all__ += group_plan.__all__
__all__ += mocks.__all__
__all__ += noise.__all__
__all__ += observation.__all__
__all__ += plan.__all__
__all__ += profile.__all__
__all__ += sky.__all__
__all__ += weights.__all__


def _register_observations() -> None:
    from gbkfit.observation.observation import (
        observation_parser as abstract_parser)
    abstract_parser.register([
        ObservationPixelBrightness,
        ObservationPixelMoments,
        ObservationPixelSpectra,
        ObservationProfile,
        ObservationRegionMoments,
        ObservationRegionSpectra,
        ObservationSlitSpectra,
        ObservationSlitlessSpectra,
        ObservationVisibilities
    ])


_register_observations()
