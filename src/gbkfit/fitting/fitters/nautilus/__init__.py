try:
    import nautilus
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "to use the nautilus fitter, please install nautilus-sampler")

from .core import *
