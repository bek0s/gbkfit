
try:
    from .dynesty import FitterDynestyDNS, FitterDynestySNS
except ImportError:
    pass

try:
    from .emcee import FitterEmcee
except ImportError:
    pass

try:
    from .pygmo import FitterPygmo
except ImportError:
    pass

from .scipy import FitterScipyLeastSquares, FitterScipyMinimize
