from .base import *
from .geometries import *
from .models import *
from .lines import *
from .points import *

from . import disks


def _register_models() -> None:
    from gbkfit.model.base import model_parser as parser
    parser.register(ModelIntensity2D)
    parser.register(ModelIntensity3D)
    parser.register(ModelKinematics2D)
    parser.register(ModelKinematics3D)


_register_models()
