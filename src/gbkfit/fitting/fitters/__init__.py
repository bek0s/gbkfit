from gbkfit.fitting.base import fitter_parser
from .cmaes import FitterCMAES
from .dnest4 import FitterDNest4
from .dynesty import FitterDynestyDynamic, FitterDynestyStatic
from .emcee import FitterEmcee
from .grid import FitterGrid
from .lmfit import FitterLMFitLeastSquares
from .multistart import FitterMultiStart
from .nautilus import FitterNautilus
from .scipy import FitterScipyLeastSquares
from .ultranest import FitterUltraNest
from .zeus import FitterZeus


# The fitters (those that need optional packages too: see their
# requires)
fitter_parser.register([
    FitterCMAES,
    FitterDNest4,
    FitterDynestyDynamic,
    FitterDynestyStatic,
    FitterEmcee,
    FitterGrid,
    FitterLMFitLeastSquares,
    FitterMultiStart,
    FitterNautilus,
    FitterScipyLeastSquares,
    FitterUltraNest,
    FitterZeus,
])
