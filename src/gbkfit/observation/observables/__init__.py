from .base import *
from .pixel_brightness import *
from .pixel_moments import *
from .pixel_spectra import *
from .region_moments import *
from .region_spectra import *
from .slit_spectra import *


def _register_observables():
    from gbkfit.observation.observables.base import (
        observable_parser as abstract_parser)
    abstract_parser.register([
        PixelBrightness,
        PixelMoments,
        PixelSpectra,
        RegionMoments,
        RegionSpectra,
        SlitSpectra
    ])


_register_observables()
