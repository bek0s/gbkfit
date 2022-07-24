
from .image import *
from .lslit import *
from .mmaps import *
from .scube import *


def _register_dmodels():
    from gbkfit.model.core import dmodel_parser as parser
    parser.register([
        DModelImage,
        DModelLSlit,
        DModelMMaps,
        DModelSCube])


_register_dmodels()
