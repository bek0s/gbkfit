
try:
    import dynesty
except ImportError:
    raise ImportError(
        "to use the dynesty fitter please install dynesty")

from .dynesty import FitterDynestyDNS, FitterDynestySNS
