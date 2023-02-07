
from .image import *
from .lslit import *
from .mmaps import *
from .scube import *


def _register_dmodels():
    from gbkfit.model.core import dmodel_parser as abstract_parser
    abstract_parser.register([
        DModelImage,
        DModelLSlit,
        DModelMMaps,
        DModelSCube])


_register_dmodels()
