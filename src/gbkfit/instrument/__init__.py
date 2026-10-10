"""
The instrument: what the telescope and the instrument do to the light of
a model. The primary beam attenuates it, the point spread function (PSF)
spreads it on the sky, and the line spread function (LSF) along the
spectral axis.
"""

from .instrument import *
from .lsfs import *
from .primary_beams import *
from .psfs import *
from .varying import *
