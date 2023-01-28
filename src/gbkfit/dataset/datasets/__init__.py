
from .image import *
from .lslit import *
from .mmaps import *
from .scube import *


def _register_datasets():
    from gbkfit.dataset.core import dataset_parser as parser
    parser.register([
        DatasetImage,
        DatasetLSlit,
        DatasetMMaps,
        DatasetSCube
    ])


_register_datasets()
