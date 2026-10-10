
from gbkfit.utils import parseutils


def _register_fitters():
    from gbkfit.fitting.base import fitter_parser as abstract_parser
    parsers = [
        'gbkfit.fitting.fitters.lmfit.FitterLMFitLeastSquares',
        'gbkfit.fitting.fitters.nautilus.FitterNautilus',
        'gbkfit.fitting.fitters.dynesty.FitterDynesty',
        'gbkfit.fitting.fitters.cmaes.FitterCMAES',
        'gbkfit.fitting.fitters.emcee.FitterEmcee',
        'gbkfit.fitting.fitters.scipy.FitterScipyLeastSquares',
        'gbkfit.fitting.fitters.multistart.FitterMultiStart',
    ]
    parseutils.register_optional_parsers(abstract_parser, parsers, 'fitter')


_register_fitters()
