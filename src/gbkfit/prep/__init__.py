"""
Data preparation: data read from FITS files made ready for the datasets
of gbkfit (cropped, masked, clipped, in the axis order and direction of
the model, and with velocity spectral axes), the binning of maps and
cubes for region datasets (and Voronoi bins of a map to a target
signal-to-noise), and the tables of the PSFs and LSFs of instruments.
The prep task of gbkfit-cli is a client of this module.
"""

from .arrays import *
from .binning import *
from .masks import *
from .moments import *
from .muse import *
from .nirspec import *

