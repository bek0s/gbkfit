
from gbkfit.utils import parseutils


def _register_fitters():
    from gbkfit.fitting.base import fitter_parser as abstract_parser
    parsers = [
        'gbkfit.fitting.fitters.lmfit.FitterLMFitLeastSquares',
        'gbkfit.fitting.fitters.nautilus.FitterNautilus',
        'gbkfit.fitting.fitters.cmaes.FitterCMAES',
        'gbkfit.fitting.fitters.emcee.FitterEmcee',
    ]
    parseutils.register_optional_parsers(abstract_parser, parsers, 'fitter')


_register_fitters()
