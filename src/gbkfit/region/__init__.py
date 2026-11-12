"""
The regions of the sky in which data were measured (see Regions and
regions_parser): bins of the pixels of a grid (RegionsBins), or
apertures (RegionsApertures), each of whose data is the sum of the
pixels it overlaps, weighted by the fraction of their area inside it
(see Aperture and aperture_parser).
"""

from . import apertures, regions
from .apertures import *
from .regions import *


__all__ = []
__all__ += apertures.__all__
__all__ += regions.__all__
