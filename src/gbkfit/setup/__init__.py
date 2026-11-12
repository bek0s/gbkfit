"""
The front door of GBKFIT: galaxies, their observations and their
parameters set up on drivers (see Setup), in Python or from a
configuration (see load).

A configuration (a YAML or JSON file, or a dictionary) has the section
'version', the version of its format (see gbkfit.config.FORMAT_VERSION;
a setup of a project has that of its project), the sections 'galaxies',
'observations' and 'params', and can have the sections 'root' (the
directory its relative paths are relative to), 'driver' or 'drivers',
'lines', 'bands', 'pdescs', 'depth_order', 'cosmology', 'fitter',
'workers', 'mock' (mock data; see Setup.mock) and 'bootstrap' (a
bootstrap of a fit; see Setup.bootstrap).
"""

from . import _config, _outputs, _setup
from ._config import *
from ._outputs import *
from ._setup import *


__all__ = []
__all__ += _config.__all__
__all__ += _outputs.__all__
__all__ += _setup.__all__
