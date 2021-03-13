
from .lsfs import *


def _register_lsfs():
    from gbkfit.psflsf.core import lsf_parser as parser
    parser.register(LSFGauss)
    parser.register(LSFGGauss)
    parser.register(LSFImage)
    parser.register(LSFLorentz)
    parser.register(LSFMoffat)


_register_lsfs()
