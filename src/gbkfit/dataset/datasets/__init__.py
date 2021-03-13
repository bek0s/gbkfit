
from .image import DatasetImage
from .lslit import DatasetLSlit
from .mmaps import DatasetMMaps
from .scube import DatasetSCube


def _register_datasets():
    from gbkfit.dataset.core import dataset_parser as parser
    parser.register(DatasetImage)
    parser.register(DatasetLSlit)
    parser.register(DatasetMMaps)
    parser.register(DatasetSCube)


_register_datasets()
