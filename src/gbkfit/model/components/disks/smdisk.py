from collections.abc import Sequence
from typing import Any, Self

import numpy as np

from gbkfit.driver import Driver
from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import gridutils
from .._units import DataUnit
from ..base import Component, ComponentPlan, FieldsLayout
from ..geometries import Geometry
from ..lines import Band, LineCatalogue
from ..roles import Emission, Opacity
from . import _detail, _disk, traits
from .parts import Density, Kinematics


__all__ = [
    'SMDisk'
]


class SmoothDisk(_disk.Disk):
    """A disk evaluated smoothly: on the pixels or voxels of its grid."""

    def plan(
            self,
            driver: Driver,
            nlines: int,
            dtype: np.dtype,
            opacity_rows: Sequence[int] | None = None
    ) -> _disk.DiskPlan:
        return SmoothDiskPlan(self, driver, nlines, dtype, opacity_rows)


class SmoothDiskPlan(_disk.DiskPlan):
    """The evaluation of a SmoothDisk."""

    def fields_layout(self) -> FieldsLayout:
        # (a thin disk is a layer where its plane crosses each line of
        # sight; a thick one fills the voxels with j, v, s and its line
        # opacity)
        if self._disk.traits('rht'):
            nk = self._fields_lines()
            return FieldsLayout(
                layer=False, nslots=3 + nk, line_opacity=nk > 0)
        return FieldsLayout(layer=True, nslots=4)

    def _impl_evaluate(
            self,
            params: ParamValues,
            grid_and_outputs: dict[str, Any],
            out_extra: dict[str, Any] | None
    ) -> None:
        self._backend.smdisk_evaluate(self._native_disk, **grid_and_outputs)


class SMDisk(Component):
    """
    A disk evaluated smoothly: at every pixel of the grid (a thin disk,
    whose light is where its plane crosses the line of sight of the pixel
    on a grid with a depth axis), or at every voxel of a grid with a depth
    axis (a thick disk, with density height traits).

    Parameters
    ----------
    density : Density
        Its density (thick with height traits).
    kinematics : Kinematics, optional
        How it moves; required to emit lines.
    emission : Emission, optional
        The light it emits.
    opacity : Opacity, optional
        What it removes from the light behind it (thick disks only).
    zptraits : ZPTrait or Sequence of ZPTrait, optional
        Its vertical distortion polar traits (e.g. a warp out of its
        plane; thick disks only).
    sptraits : SPTrait or Sequence of SPTrait, optional
        Its selection polar traits: the parts of the disk that are kept
        (e.g. a range of azimuths), for all its roles.
    loose : bool, optional
        Whether its centre and systemic velocity are node-wise (a value
        for each ring); by default, not, or those of its geometry.
    tilted : bool, optional
        Whether its position angle and inclination are node-wise; by
        default, not, or those of its geometry.
    rnmin, rnmax, rnsep, rnlen, rnodes : optional
        The radii of its rings (arcsec): rnodes; or rnmin, rnmax and rnsep
        (every rnsep); or rnmin, rnmax and rnlen (rnlen rings). Those of
        its geometry if it is warped.
    rstep : float, optional
        The width of the rings it is evaluated on, at most half the
        smallest separation of its radii; by default that, or 1 if less.
    interp : str, optional
        How its node-wise parameters are interpolated between its radii:
        'linear', 'akima' or 'pchip'.
    geometry : Geometry, optional
        A geometry that it shares with other components (see Geometry).
    name : str, optional
        Its name (see Component).

    Raises
    ------
    ConfigError
        If the options are not valid (e.g. the rings), conflict with its
        geometry, or its parts and roles do not go together (e.g. lines
        without kinematics, or opacity or zptraits on a thin disk).
    """

    @staticmethod
    def type() -> str:
        return 'smdisk'

    @classmethod
    def load(
            cls,
            info: dict[str, Any],
            catalogue: LineCatalogue | None = None
    ) -> Self:
        return cls(**_detail.load_disk_component(cls, info, catalogue))

    def dump(self) -> dict[str, Any]:
        return _detail.dump_disk_component(
            self, self._disk, self._density, self._kinematics)

    def __init__(
            self,
            density: Density,
            kinematics: Kinematics | None = None,
            emission: Emission | None = None,
            opacity: Opacity | None = None,
            zptraits: traits.ZPTrait | Sequence[traits.ZPTrait] | None = None,
            sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None = None,
            loose: bool | None = None,
            tilted: bool | None = None,
            rnmin: float | None = None,
            rnmax: float | None = None,
            rnsep: float | None = None,
            rnlen: int | None = None,
            rnodes: Sequence[float] | None = None,
            rstep: float | None = None,
            interp: str = 'linear',
            geometry: Geometry | None = None,
            name: str | None = None
    ):
        super().__init__(name, geometry, emission, opacity)
        self._density = density
        self._kinematics = kinematics
        self._disk = _detail.make_disk(
            SmoothDisk, density, kinematics, emission, opacity, zptraits,
            sptraits, geometry=geometry, loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp)
        self._lines = _detail.make_lines(self._disk, emission)
        self._bands = _detail.make_bands(self._disk, emission)

    def density(self) -> Density:
        """Return its density."""
        return self._density

    def kinematics(self) -> Kinematics | None:
        """Return its kinematics, if any."""
        return self._kinematics

    def thick(self) -> bool:
        return self._density.thick()

    def pdescs(self) -> dict[str, ParamDesc]:
        lines = self._lines.pdescs() if self._lines is not None else {}
        bands = self._bands.pdescs() if self._bands is not None else {}
        return self._disk.pdescs() | lines | bands

    def constants(self) -> dict[str, Any]:
        return dict(
            rnodes=self._disk.rnodes(), subrnodes=self._disk.subrnodes())

    def circular_velocity_params(self) -> dict[str, tuple[float, ...]]:
        return self._disk.circular_velocity_params()

    def density_disk(self) -> _disk.Disk:
        return self._disk

    def plan(
            self,
            driver: Driver,
            spectral: gridutils.Grid,
            dtype: np.dtype,
            lines: tuple[str, ...] | None,
            band: Band | None,
            emits: bool,
            opacity_ratios: np.ndarray | None,
            data_unit: DataUnit | None,
            redshift: float
    ) -> ComponentPlan:
        emission = self.emission()
        return _detail.disk_plan(
            self._disk, self._lines, self._bands,
            emission.unit() if emission is not None else None, driver,
            spectral, dtype, lines, band, emits, opacity_ratios, data_unit,
            redshift)
