"""
The instrument: what the telescope and the instrument do to the light of
a model. The primary beam attenuates it, the point spread function (PSF)
spreads it on the sky, and the line spread function (LSF) along the
spectral axis. Slitless spectrographs add the slit mask in front of their
disperser (slit_masks) and the trace of its spectra on the detector
(traces).
"""

from .instrument import *
from .lsfs import *
from .primary_beams import *
from .psfs import *
from .slit_masks import *
from .traces import *
from .varying import *
