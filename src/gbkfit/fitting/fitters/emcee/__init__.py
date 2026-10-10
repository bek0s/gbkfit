try:
    import emcee
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "the emcee fitter needs the optional package emcee: "
        "pip install 'gbkfit[emcee]'") from e

from .core import *
