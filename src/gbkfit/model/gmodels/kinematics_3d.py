
from collections.abc import Sequence

from gbkfit.model.core import GModelSCube
from gbkfit.utils import parseutils
from ._gmodel import GModel3D
from .core import OpacityComponent3D, SpectralComponent3D
from .opacity_mcdisk_3d import OpacityMCDisk3D
from .opacity_smdisk_3d import OpacitySMDisk3D
from .spectral_mcdisk_3d import SpectralMCDisk3D
from .spectral_smdisk_3d import SpectralSMDisk3D


__all__ = [
    'GModelKinematics3D'
]


class GModelKinematics3D(GModel3D, GModelSCube):

    _cmp_parser = parseutils.TypedParser(SpectralComponent3D, [
        SpectralMCDisk3D,
        SpectralSMDisk3D])

    _ocmp_parser = parseutils.TypedParser(OpacityComponent3D, [
        OpacityMCDisk3D,
        OpacitySMDisk3D])

    @staticmethod
    def type():
        return 'kinematics_3d'

    def __init__(
            self,
            components:
            SpectralComponent3D | Sequence[SpectralComponent3D],
            opacity_components:
            OpacityComponent3D | Sequence[OpacityComponent3D] | None = None,
            size_z: int | None = None,
            step_z: int | float | None = None,
            zero_z: int | float | None = None
    ):
        super().__init__(
            components, opacity_components, size_z, step_z, zero_z)

    def evaluate_scube(
            self, driver, params, scube, weights, size, step, zero, rota,
            dtype, out_extra):
        self._evaluate(
            driver, params, dict(scube=scube), (size[2], step[2], zero[2]),
            weights, size, step, zero, rota, dtype, out_extra)
