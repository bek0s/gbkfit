
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
            log.warning(
                f"could not register factory for type {factory}: {str(e)}")


def _register_drivers():
    from gbkfit.driver.driver import parser
    factories = [
        'gbkfit.driver.drivers.cuda.driver.DriverCUDA',
        'gbkfit.driver.drivers.host.driver.DriverHost']
    _register_factories(parser, factories)


def _register_fitters():
    from gbkfit.fitting.fitter import parser
    factories = [
        'gbkfit.fitting.fitters.dynesty.FitterDyneestyDNS',
        'gbkfit.fitting.fitters.dynesty.FitterDynestySNS',
        'gbkfit.fitting.fitters.emcee.FitterEmcee',
        'gbkfit.fitting.fitters.lmfit.FitterLMFitLeastSquares',
        'gbkfit.fitting.fitters.lmfit.FitterLMFitNelderMead',
        'gbkfit.fitting.fitters.pygmo.FitterPygmo',
        'gbkfit.fitting.fitters.pyswarms.FitterPySwarmsGeneral',
        'gbkfit.fitting.fitters.pyswarms.FitterPySwarmsGlobal',
        'gbkfit.fitting.fitters.pyswarms.FitterPySwarmsLocal',
        'gbkfit.fitting.fitters.scipy.FitterScipyLeastSquares',
        'gbkfit.fitting.fitters.scipy.FitterScipyMinimize']
    _register_factories(parser, factories)


def _register_datasets():
    from gbkfit.dataset.dataset import parser
    from gbkfit.dataset.datasets import (
        DatasetImage, DatasetLSlit, DatasetMMaps, DatasetSCube)
    parser.register(DatasetImage)
    parser.register(DatasetLSlit)
    parser.register(DatasetMMaps)
    parser.register(DatasetSCube)


def _register_dmodels():
    from gbkfit.model.dmodel import parser
    from gbkfit.model.dmodels import (
        DModelImage, DModelLSlit, DModelMMaps, DModelSCube)
    parser.register(DModelImage)
    parser.register(DModelLSlit)
    parser.register(DModelMMaps)
    parser.register(DModelSCube)


def _register_gmodels():
    from gbkfit.model.gmodel import parser
    from gbkfit.model.gmodels import (
        GModelIntensity2D, GModelIntensity3D,
        GModelKinematics2D, GModelKinematics3D)
    parser.register(GModelIntensity2D)
    parser.register(GModelIntensity3D)
    parser.register(GModelKinematics2D)
    parser.register(GModelKinematics3D)


def _register_psfs():
    from gbkfit.psflsf.psflsf import psf_parser
    from gbkfit.psflsf.psflsfs.psfmodels import (
        PSFGauss, PSFGGauss, PSFImage, PSFLorentz, PSFMoffat)
    psf_parser.register(PSFGauss)
    psf_parser.register(PSFGGauss)
    psf_parser.register(PSFImage)
    psf_parser.register(PSFLorentz)
    psf_parser.register(PSFMoffat)


def _register_lsfs():
    from gbkfit.psflsf.psflsf import lsf_parser
    from gbkfit.psflsf.psflsfs.lsfmodels import (
        LSFGauss, LSFGGauss, LSFImage, LSFLorentz, LSFMoffat)
    lsf_parser.register(LSFGauss)
    lsf_parser.register(LSFGGauss)
    lsf_parser.register(LSFImage)
    lsf_parser.register(LSFLorentz)
    lsf_parser.register(LSFMoffat)


def _init():
    _register_datasets()
    _register_drivers()
    _register_dmodels()
    _register_gmodels()
    _register_psfs()
    _register_lsfs()
    _register_fitters()


_init()
