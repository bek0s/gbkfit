
from .core import *

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


component_d2d_parser.register(DensitySMDisk2D)
component_d3d_parser.register(DensityMCDisk3D)
component_d3d_parser.register(DensitySMDisk3D)
component_s2d_parser.register(SpectralSMDisk2D)
component_s3d_parser.register(SpectralMCDisk3D)
component_s3d_parser.register(SpectralSMDisk3D)
