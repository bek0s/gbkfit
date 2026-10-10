from .base import *
from .data import *
from .pixel_brightness import *
from .pixel_moments import *
from .pixel_spectra import *
from .region_moments import *
from .region_spectra import *
from .slit_spectra import *


def _register_datasets() -> None:
    from .base import dataset_parser as abstract_parser
    abstract_parser.register([
        DatasetPixelBrightness,
        DatasetPixelMoments,
        DatasetPixelSpectra,
        DatasetRegionMoments,
        DatasetRegionSpectra,
        DatasetSlitSpectra
    ])


_register_datasets()
