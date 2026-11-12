"""
Datasets: the data of an observation and where they were measured (see
Dataset and dataset_parser), one type for each kind of measurement:
images, moment maps and spectral cubes on a grid of pixels, the moments
and spectra of regions, the spectra of slits, slitless spectra,
interferometric visibilities, and profiles at radii. Each holds named
data items (see Data), and reads them from files; describe_fits says
what the HDUs of a FITS file hold before a dataset is made of them.
"""

from . import (
    base, data, describe, pixel_brightness, pixel_moments, pixel_spectra,
    profile, region_moments, region_spectra, slit_spectra,
    slitless_spectra, visibilities)
from .base import *
from .data import *
from .describe import *
from .pixel_brightness import *
from .pixel_moments import *
from .pixel_spectra import *
from .profile import *
from .region_moments import *
from .region_spectra import *
from .slit_spectra import *
from .slitless_spectra import *
from .visibilities import *


__all__ = []
__all__ += base.__all__
__all__ += data.__all__
__all__ += describe.__all__
__all__ += pixel_brightness.__all__
__all__ += pixel_moments.__all__
__all__ += pixel_spectra.__all__
__all__ += profile.__all__
__all__ += region_moments.__all__
__all__ += region_spectra.__all__
__all__ += slit_spectra.__all__
__all__ += slitless_spectra.__all__
__all__ += visibilities.__all__


def _register_datasets() -> None:
    from .base import dataset_parser as abstract_parser
    abstract_parser.register([
        DatasetPixelBrightness,
        DatasetPixelMoments,
        DatasetPixelSpectra,
        DatasetProfile,
        DatasetRegionMoments,
        DatasetRegionSpectra,
        DatasetSlitSpectra,
        DatasetSlitlessSpectra,
        DatasetVisibilities
    ])


_register_datasets()
