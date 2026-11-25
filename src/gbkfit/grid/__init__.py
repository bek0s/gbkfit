"""
Grids of pixels with world coordinates (see Grid and Coords), their
spectral axes (see SpectralAxis and VelocityConvention), and the types of
the options of grids on the sky in configurations (see options).
"""

from . import coords, options, spectral
from .coords import *
from .options import *
from .spectral import *


__all__ = []
__all__ += coords.__all__
__all__ += options.__all__
__all__ += spectral.__all__
