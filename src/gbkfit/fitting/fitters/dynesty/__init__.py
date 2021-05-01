
try:
    import dynesty
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "to use the dynesty fitter, please install dynesty")

from .dynesty import *
