from .base import *
from .gmodels import *
from .lines import *
from .points import *

from . import disks


def _register_gmodels():
    from gbkfit.model.base import gmodel_parser as parser
    parser.register(GModelIntensity2D)
    parser.register(GModelIntensity3D)
    parser.register(GModelKinematics2D)
    parser.register(GModelKinematics3D)


_register_gmodels()
