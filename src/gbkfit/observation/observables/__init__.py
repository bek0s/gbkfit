from gbkfit.utils import parseutils
from .core import *
from .pixel_brightness import *
from .pixel_moments import *
from .pixel_spectra import *
from .region_moments import *
from .region_spectra import *
from .slit_spectra import *


observable_parser = parseutils.TypedParser(Observable, [
    PixelBrightness,
    PixelMoments,
    PixelSpectra,
    RegionMoments,
    RegionSpectra,
    SlitSpectra])
