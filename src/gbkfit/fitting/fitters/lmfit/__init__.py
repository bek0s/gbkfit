
try:
    import lmfit
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "to use the lmfit fitter, please install lmfit")

from .fitter import *
