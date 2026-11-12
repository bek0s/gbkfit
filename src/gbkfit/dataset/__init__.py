"""
Datasets: the data of an observation and where they were measured (see
Dataset and dataset_parser), one type for each kind of measurement:
images, moment maps and spectral cubes on a grid of pixels, the moments
and spectra of regions, the spectra of slits, slitless spectra,
interferometric visibilities, and profiles at radii. Each holds named
data items (see Data), and reads them from files (gbkfit.io.describe_fits
says what the HDUs of a FITS file hold before a dataset is made of them).

The datasets of regions say in which regions of the sky their data were
measured (see Regions and regions_parser): bins of the pixels of a grid
(RegionsBins), or apertures (RegionsApertures), each of whose data is
the sum of the pixels it overlaps, weighted by the fraction of their
area inside it (see Aperture and aperture_parser).
"""

from . import (
    apertures, base, data, pixel_brightness, pixel_moments,
    pixel_spectra, profile, region_moments, region_spectra, regions,
    slit_spectra, slitless_spectra, visibilities)
from .apertures import *
from .base import *
from .data import *
from .pixel_brightness import *
from .pixel_moments import *
from .pixel_spectra import *
from .profile import *
from .region_moments import *
from .region_spectra import *
from .regions import *
from .slit_spectra import *
from .slitless_spectra import *
from .visibilities import *


__all__ = []
__all__ += apertures.__all__
__all__ += base.__all__
__all__ += data.__all__
__all__ += pixel_brightness.__all__
__all__ += pixel_moments.__all__
__all__ += pixel_spectra.__all__
__all__ += profile.__all__
__all__ += region_moments.__all__
__all__ += region_spectra.__all__
__all__ += regions.__all__
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
