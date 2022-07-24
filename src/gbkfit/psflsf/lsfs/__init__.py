
from .lsfs import *


def _register_lsfs():
    from gbkfit.psflsf.core import lsf_parser as parser
    parser.register([
        LSFPoint,
        LSFGauss,
        LSFGGauss,
        LSFLorentz,
        LSFMoffat,
        LSFImage
    ])


_register_lsfs()
