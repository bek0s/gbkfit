
import importlib
import logging


_log = logging.getLogger(__name__)


def _register_factories(parser, factories):
    for factory in factories:
        try:
            mod_name = factory.rsplit('.', 1)[0]
            cls_name = factory.rsplit('.', 1)[1]
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            parser.register(cls)
        except Exception as e:
            _log.warning(
                f"could not register fitter factory {factory}; "
                f"{e.__class__.__name__}: {e}")


def _register_fitters():
    from gbkfit.fitting.core import fitter_parser as parser
    factories = [
        # dynesty
        'gbkfit.fitting.fitters.dynesty.FitterDynestyDNS',
        'gbkfit.fitting.fitters.dynesty.FitterDynestySNS',
        # emcee
        'gbkfit.fitting.fitters.emcee.FitterEmcee',
        # lmfit
        'gbkfit.fitting.fitters.lmfit.FitterLMFitLeastSquares',
        'gbkfit.fitting.fitters.lmfit.FitterLMFitNelderMead',
        # pygmo
        'gbkfit.fitting.fitters.pygmo.FitterPygmoDE',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoSADE',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoDE1220',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoPSO',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoCMAES',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoXNES',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoIpopt',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoNLopt',
        'gbkfit.fitting.fitters.pygmo.FitterPygmoScipy']
    _register_factories(parser, factories)


_register_fitters()
