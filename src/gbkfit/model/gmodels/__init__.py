
from .intensity_2d import GModelIntensity2D
from .intensity_3d import GModelIntensity3D
from .kinematics_2d import GModelKinematics2D
from .kinematics_3d import GModelKinematics3D

from .density_mcdisk_3d import DensityMCDisk3D
from .density_smdisk_2d import DensitySMDisk2D
from .density_smdisk_3d import DensitySMDisk3D
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
