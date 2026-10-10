try:
    import cma
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "the cmaes fitter needs the optional package cma: "
        "pip install 'gbkfit[cmaes]'") from e

from .core import *
