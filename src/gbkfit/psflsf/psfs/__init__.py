
from .psfs import *


def _register_psfs():
    from gbkfit.psflsf.core import psf_parser as parser
    parser.register([
        PSFPoint,
        PSFGauss,
        PSFGGauss,
        PSFLorentz,
        PSFMoffat,
        PSFImage,
    ])


_register_psfs()
