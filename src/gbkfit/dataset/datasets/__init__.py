
from .region_spectra import *
from .region_moments import *
from .pixel_brightness import *
from .slit_spectra import *
from .pixel_moments import *
from .pixel_spectra import *


def _register_datasets():
    from gbkfit.dataset.core import dataset_parser as abstract_parser
    abstract_parser.register([
        DatasetRegionSpectra,
        DatasetRegionMoments,
        DatasetPixelBrightness,
        DatasetSlitSpectra,
        DatasetPixelMoments,
        DatasetPixelSpectra
    ])


_register_datasets()
