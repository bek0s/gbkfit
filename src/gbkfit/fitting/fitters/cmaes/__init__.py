try:
    import cma
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "to use the cmaes fitter, please install cma")

from .core import *
