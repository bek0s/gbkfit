
try:
    import pymoo
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "to use the pymoo fitter, please install pymoo")

from .core import *
from .fitters import *
