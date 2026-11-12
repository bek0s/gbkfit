"""
The model of galaxies: galaxies (see Galaxy and galaxy_parser) made of
components that emit light or absorb it (thin and thick disks, cones,
spheroids, points and particles; see gbkfit.model.components), their
mass and its gravity (see gbkfit.model.mass and gbkfit.model.gravity),
and their distance (see distance). Scenes draw galaxies together on the
grid of an image or a spectral cube (see scenes), and an evaluation also
gives what it computes on the way (see extras). The model never imports
the datasets: observations (see gbkfit.observation) turn it into the
data of a dataset.
"""

from ._functions import GalaxyFunctions
from ._galaxy_stage import GalaxyStage, GalaxyState
from .distance import *
from .extras import *
from .galaxy import *
from .gravity import *
from .mass import *
from .scenes import *

from . import components
from .components import *
