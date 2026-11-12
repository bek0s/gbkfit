"""
The observables: what a dataset measures, made from a model seen
through an instrument (see Observable and observable_parser), one for
each kind of dataset: images, moment maps and spectral cubes on a grid
of pixels (PixelBrightness, PixelMoments, PixelSpectra), the moments
and spectra of regions (RegionMoments, RegionSpectra), the spectra of
slits (SlitSpectra), slitless spectra (SlitlessSpectra), and
interferometric visibilities (Visibilities). gbkfit.observation exports
them too.
"""

from . import (
    base, pixel_brightness, pixel_moments, pixel_spectra, region_moments,
    region_spectra, slit_spectra, slitless_spectra, visibilities)
from .base import *
from .pixel_brightness import *
from .pixel_moments import *
from .pixel_spectra import *
from .region_moments import *
from .region_spectra import *
from .slit_spectra import *
from .slitless_spectra import *
from .visibilities import *


__all__ = []
__all__ += base.__all__
__all__ += pixel_brightness.__all__
__all__ += pixel_moments.__all__
__all__ += pixel_spectra.__all__
__all__ += region_moments.__all__
__all__ += region_spectra.__all__
__all__ += slit_spectra.__all__
__all__ += slitless_spectra.__all__
__all__ += visibilities.__all__


def _register_observables() -> None:
    from gbkfit.observation.observables.base import (
        observable_parser as abstract_parser)
    abstract_parser.register([
        PixelBrightness,
        PixelMoments,
        PixelSpectra,
        RegionMoments,
        RegionSpectra,
        SlitSpectra,
        SlitlessSpectra,
        Visibilities
    ])


_register_observables()
