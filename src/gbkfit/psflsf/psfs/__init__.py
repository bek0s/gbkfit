
from .psfs import *


def _register_psfs():
    from gbkfit.psflsf.core import psf_parser as parser
    parser.register(PSFGauss)
    parser.register(PSFGGauss)
    parser.register(PSFImage)
    parser.register(PSFLorentz)
    parser.register(PSFMoffat)


_register_psfs()
