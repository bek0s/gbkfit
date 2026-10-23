from .base import *
from .data import *
from .pixel_brightness import *
from .pixel_moments import *
from .pixel_spectra import *
from .profile import *
from .region_moments import *
from .region_spectra import *
from .slit_spectra import *
from .slitless_spectra import *
from .visibilities import *


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
