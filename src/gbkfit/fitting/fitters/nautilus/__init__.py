try:
    import nautilus
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "the nautilus fitter needs the optional package nautilus-sampler: "
        "pip install 'gbkfit[nautilus]'") from e

from .core import *
