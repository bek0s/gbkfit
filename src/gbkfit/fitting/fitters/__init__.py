
from gbkfit.utils import parseutils


def _register_fitters():
    from gbkfit.fitting.core import fitter_parser as abstract_parser
    parsers = [
        # dynesty
        'gbkfit.fitting.fitters.dynesty.FitterDynestyDNS',
        'gbkfit.fitting.fitters.dynesty.FitterDynestySNS',
        # emcee
        'gbkfit.fitting.fitters.emcee.FitterEmcee',
        # lmfit
        'gbkfit.fitting.fitters.lmfit.FitterLMFitLeastSquares',
        'gbkfit.fitting.fitters.lmfit.FitterLMFitNelderMead',
        # pymoo
        'gbkfit.fitting.fitters.pymoo.FitterPymooNSGA2',
        'gbkfit.fitting.fitters.pymoo.FitterPymooPSO',
        # pygmo
        'gbkfit.fitting.fitters.pygmo.FitterPygmoGACO',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoDE',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoSADE',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoDE1220',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoGWO',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoIHS',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoPSO',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoGPSO',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoSEA',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoSGA',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoSA',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoABC',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoCMAES',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoXNES',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoNSGA2',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoMOEAD',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoGMOEAD',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoMACO',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoNSPSO',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoCompassSearch',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoNLopt',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoScipy',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoIpopt',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoSNOPT',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoWORHP',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoMBH'
    ]
    parseutils.register_optional_parsers(abstract_parser, parsers, 'fitter')


_register_fitters()
