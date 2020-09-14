
try:
    import emcee as _emcee
except ImportError:
    raise ModuleNotFoundError(
        "to use the emcee fitter please install emcee")

from .core import *
from .moves import *
from .sampler import *
