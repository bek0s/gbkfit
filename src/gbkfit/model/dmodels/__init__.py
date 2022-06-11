
from .image import DModelImage
from .lslit import DModelLSlit
from .mmaps import DModelMMaps
from .scube import DModelSCube


def _register_dmodels():
    from gbkfit.model.core import dmodel_parser as parser
    parser.register([
        DModelImage,
        DModelLSlit,
        DModelMMaps,
        DModelSCube])


_register_dmodels()
