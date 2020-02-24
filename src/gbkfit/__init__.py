
import importlib
import logging

from gbkfit._version import __version__


log = logging.getLogger(__name__)


def _register_factories(parser, factories):
    for factory in factories:
        try:
            mod_name = factory.rsplit('.', 1)[0]
            cls_name = factory.rsplit('.', 1)[1]
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            parser.register(cls)
        except (AttributeError, ImportError) as e:
            log.warning(f"Could not load factory {factory}: {str(e)}")


def _register_brokers():
    from gbkfit.broker import parser
    factories = [
        'gbkfit.broker.brokers.dask.BrokerDask',
        'gbkfit.broker.brokers.ray.BrokerRay']
    _register_factories(parser, factories)


def _register_drivers():
    from gbkfit.driver import parser
    factories = [
        'gbkfit.driver.drivers.cuda.driver.DriverCUDA',
        'gbkfit.driver.drivers.host.driver.DriverHost']
    _register_factories(parser, factories)


def _register_fitters():
    from gbkfit.fitter import parser
    factories = [
        'gbkfit.fitter.fitters.dynesty.FitterDynestyDNestedSampling',
        'gbkfit.fitter.fitters.dynesty.FitterDynestySNestedSampling',
        'gbkfit.fitter.fitters.scipy.FitterScipyLeastSquares',
        'gbkfit.fitter.fitters.pygmo.FitterPygmo']
    _register_factories(parser, factories)


def _register_dmodels():
    from gbkfit.dmodel import parser
    from gbkfit.dmodel.dmodels import (
        DModelImage, DModelLSlit, DModelMMaps, DModelSCube)
    parser.register(DModelImage)
    parser.register(DModelLSlit)
    parser.register(DModelMMaps)
    parser.register(DModelSCube)


def _register_gmodels():
    from gbkfit.gmodel import parser
    from gbkfit.gmodel.gmodels import (
        GModelIntensity2D, GModelIntensity3D,
        GModelKinematics2D, GModelKinematics3D)
    parser.register(GModelIntensity2D)
    parser.register(GModelIntensity3D)
    parser.register(GModelKinematics2D)
    parser.register(GModelKinematics3D)


def _register_psfs():
    from gbkfit.psflsf import psf_parser
    from gbkfit.psflsf.psflsfs.psfmodels import (
        PSFGauss, PSFGGauss, PSFImage, PSFLorentz, PSFMoffat)
    psf_parser.register(PSFGauss)
    psf_parser.register(PSFGGauss)
    psf_parser.register(PSFImage)
    psf_parser.register(PSFLorentz)
    psf_parser.register(PSFMoffat)


def _register_lsfs():
    from gbkfit.psflsf import lsf_parser
    from gbkfit.psflsf.psflsfs.lsfmodels import (
        LSFGauss, LSFGGauss, LSFImage, LSFLorentz, LSFMoffat)
    lsf_parser.register(LSFGauss)
    lsf_parser.register(LSFGGauss)
    lsf_parser.register(LSFImage)
    lsf_parser.register(LSFLorentz)
    lsf_parser.register(LSFMoffat)


def init():
    _register_brokers()
    _register_drivers()
    _register_dmodels()
    _register_gmodels()
    _register_psfs()
    _register_lsfs()
    _register_fitters()
