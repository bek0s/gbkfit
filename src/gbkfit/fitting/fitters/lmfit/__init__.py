try:
    import lmfit
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "the lmfit fitter needs the optional package lmfit: "
        "pip install 'gbkfit[lmfit]'") from e

from .core import *
