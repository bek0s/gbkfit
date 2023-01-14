
from .intensity_2d import GModelIntensity2D
from .intensity_3d import GModelIntensity3D
from .kinematics_2d import GModelKinematics2D
from .kinematics_3d import GModelKinematics3D

from .brightness_mcdisk_3d import BrightnessMCDisk3D
from .brightness_smdisk_2d import BrightnessSMDisk2D
from .brightness_smdisk_3d import BrightnessSMDisk3D
from .opacity_mcdisk_3d import OpacityMCDisk3D
from .opacity_smdisk_3d import OpacitySMDisk3D
from .spectral_mcdisk_3d import SpectralMCDisk3D
from .spectral_smdisk_2d import SpectralSMDisk2D
from .spectral_smdisk_3d import SpectralSMDisk3D


def _register_gmodels():
    from gbkfit.model.core import gmodel_parser as parser
    parser.register(GModelIntensity2D)
    parser.register(GModelIntensity3D)
    parser.register(GModelKinematics2D)
    parser.register(GModelKinematics3D)


_register_gmodels()
