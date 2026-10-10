try:
    import dynesty
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "the dynesty fitter needs the optional package dynesty: "
        "pip install 'gbkfit[dynesty]'") from e

from .core import *
