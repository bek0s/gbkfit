try:
    import zeus
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "the zeus fitter needs the optional package zeus-mcmc: "
        "pip install 'gbkfit[zeus]'") from e

from .core import *
