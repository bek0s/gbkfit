"""
Gravitational lensing of the light of a model, in front of the telescope
(see Foreground): a lens (lens), whose masses each deflect the light of
the source plane (masses).
"""

from . import lens, masses
from .lens import *
from .masses import *


__all__ = []
__all__ += lens.__all__
__all__ += masses.__all__
