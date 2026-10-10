
from gbkfit.utils import parseutils


def _register_fitters():
    from gbkfit.fitting.base import fitter_parser as abstract_parser
    parsers = [
        'gbkfit.fitting.fitters.lmfit.FitterLMFitLeastSquares',
    ]
    parseutils.register_optional_parsers(abstract_parser, parsers, 'fitter')


_register_fitters()
