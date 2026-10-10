try:
    import ultranest
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "to use the ultranest fitter, please install ultranest")

from .core import *
