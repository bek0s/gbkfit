from ._group_plan import *
from .foreground import *
from .group import *
from .lens_masses import *
from .likelihood import *
from .line_of_sight import *
from .noise import *
from .observables import *
from .observation import *
from .profile import *
from .sky import *
from .weights import *


def _register_observations():
    from gbkfit.observation.observation import (
        observation_parser as abstract_parser)
    abstract_parser.register([
        ObservationProfile,
        ObservationSky
    ])


_register_observations()
