try:
    import eryn
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "the eryn fitter needs the optional package eryn: "
        "pip install 'gbkfit[eryn]'") from e

from .core import *
