
from collections.abc import Sequence

from gbkfit.model.core import GModelSCube
from gbkfit.utils import parseutils
from ._gmodel import ComponentGModel
from .core import SpectralComponent2D
from .spectral_smdisk_2d import SpectralSMDisk2D


__all__ = [
    'GModelKinematics2D'
]


class GModelKinematics2D(ComponentGModel, GModelSCube):

    _cmp_parser = parseutils.TypedParser(SpectralComponent2D, [
        SpectralSMDisk2D])

    @staticmethod
    def type():
        return 'kinematics_2d'

    def __init__(
            self,
            components: SpectralComponent2D | Sequence[SpectralComponent2D]
    ):
        super().__init__(components)

    def evaluate_scube(
            self, driver, params, scube, weights, size, step, zero, rota,
            dtype, out_extra):
        self._evaluate(
            driver, params, dict(scube=scube), weights,
            size, step, zero, rota, dtype, out_extra)
