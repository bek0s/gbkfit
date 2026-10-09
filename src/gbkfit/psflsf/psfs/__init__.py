
from .psfs import *


def _register_psfs():
    from gbkfit.psflsf.base import psf_parser as parser
    parser.register([
        PSFPoint,
        PSFGauss,
        PSFGGauss,
        PSFMoffat,
        PSFImage,
        PSFSum,
        PSFConvolution,
        PSFBeam
    ])


_register_psfs()
