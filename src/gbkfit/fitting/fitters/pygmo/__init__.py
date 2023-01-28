
try:
    import pygmo
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "to use the pygmo fitter, please install pygmo")

from .fitters import *
