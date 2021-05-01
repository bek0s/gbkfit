
try:
    import emcee
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "to use the emcee fitter, please install emcee")

from .core import *
from .moves import *
from .sampler import *
