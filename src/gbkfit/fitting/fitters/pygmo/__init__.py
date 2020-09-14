
try:
    import pygmo
except ImportError:
    raise ImportError(
        "could not import pygmo fitter; make sure pygmo is installed")

from .fitter import *
