"""
The disk component: a disk of rings with traits, smooth or made of
clouds (see Clouds).
"""

from collections.abc import Sequence
from typing import Annotated, Any, Literal, Self

import numpy as np

from gbkfit.driver import Driver
from gbkfit.params import ParamDesc
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError, check_options
from gbkfit.utils.typeutils import Unit
from ...mass import DiskDensity
from .. import traits
from .._needs import Need
from .._nodal import INTERPOLATIONS
from .._native import make_bands, make_lines
from .._units import DataUnit
from ..base import Component, ComponentPlan, FrameParams
from ..geometries import Geometry
from ..lines import Band, LineCatalogue
from ..roles import STARS, Emission, Opacity
from . import _detail, _disk, _stage
from .clouds import Clouds
from ..parts import Density, DiskTraitsKinematics


__all__ = [
    'Disk'
]


class Disk(Component):
    """
    A disk of rings, whose properties are the traits of its parts (see
    traits). A smooth disk is evaluated at every pixel of the grid (a
    thin disk, whose light is where its plane crosses the line of sight
    of the pixel on a grid with a depth axis), or at every voxel of a grid
    with a depth axis (a thick disk, with density height traits). A disk
    of clouds is a sample of the smooth one: clouds drawn at random from
    its density (see Clouds; thick disks only), on a grid with a depth
    axis.

    Parameters
    ----------
    density : Density
        Its density (thick with height traits).
    kinematics : DiskTraitsKinematics, optional
        How it moves (with its vertical equilibrium, if any); required to
        emit lines.
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
        its geometry if it is warped. The disk ends at its first and last
        radii, and is evaluated at the centres of the pixels: at an edge
        inside the data where its density is not about 0 (e.g. an
        exponential cut at three scale lengths), pixels switch on and off
        as its geometry changes, so that the chi-square jumps and the
        covariance errors come out too small (5 times, in a test of
        visibilities); place its edges beyond the data, or where its
        density is about 0.
    rstep : float, optional
        The width of the rings it is evaluated on, at most half the
        smallest separation of its radii; by default that, or 1 if less.
    interp : str, optional
        How its node-wise parameters are interpolated between its radii:
        'linear', 'akima' or 'pchip'.
    clouds : Clouds, optional
        Its clouds, if it is made of clouds; by default, it is smooth.
    geometry : Geometry, 'galaxy' or 'own', optional
        The geometry that it shares: that of its galaxy (by default; a
        warped one itself), a named one, or none ('own'; see
        Component).
    name : str, optional
        Its name (see Component).

    Raises
    ------
    ConfigError
        If the options are not valid (e.g. the rings), conflict with its
        geometry, or its parts and roles do not go together (e.g. lines
        without kinematics, or opacity, zptraits or clouds on a thin
        disk).
    """

    @staticmethod
    def type() -> str:
        return 'disk'

    @classmethod
    def load(
            cls,
            info: dict[str, Any],
            catalogue: LineCatalogue | None = None
    ) -> Self:
        info = dict(info)
        if info.get('clouds') is not None:
            info['clouds'] = parseutils.load_option(
                Clouds.load, info, 'clouds')
        return cls(**_detail.load_disk_component(cls, info, catalogue))

    def dump(self) -> dict[str, Any]:
        info = _detail.dump_disk_component(
            self, self._disk, self._density, self._kinematics)
        if self._clouds is not None:
            info['clouds'] = self._clouds.dump()
        return info

    def __init__(
            self,
            density: Density,
            kinematics: DiskTraitsKinematics | None = None,
            emission: Emission | None = None,
            opacity: Opacity | None = None,
            zptraits: traits.ZPTrait | Sequence[traits.ZPTrait] | None = None,
            sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None = None,
            loose: bool | None = None,
            tilted: bool | None = None,
            rnmin: Annotated[float, Unit('arcsec')] | None = None,
            rnmax: Annotated[float, Unit('arcsec')] | None = None,
            rnsep: Annotated[float, Unit('arcsec')] | None = None,
            rnlen: int | None = None,
            rnodes: Annotated[Sequence[float], Unit('arcsec')] | None = None,
            rstep: Annotated[float, Unit('arcsec')] | None = None,
            interp: Literal[INTERPOLATIONS] = 'linear',
            clouds: Clouds | None = None,
            geometry: Geometry | Literal['galaxy', 'own'] = 'galaxy',
            name: str | None = None
    ):
        check_options(Disk.__init__, locals())
        super().__init__(name, geometry, emission, opacity)
        if clouds is not None and not density.thick():
            raise ConfigError(
                "a disk of clouds needs density height traits (it is "
                "thick)")
        self._density = density
        self._kinematics = kinematics
        self._clouds = clouds
        self._disk = _detail.make_disk(
            density, kinematics, emission, opacity, zptraits, sptraits,
            geometry=geometry if isinstance(geometry, Geometry) else None,
            loose=loose, tilted=tilted, rnmin=rnmin,
            rnmax=rnmax, rnsep=rnsep, rnlen=rnlen, rnodes=rnodes,
            rstep=rstep, interp=interp, clouds=clouds)
        self._lines = make_lines(self._disk.pdescs(), emission)
        self._bands = make_bands(self._disk.pdescs(), emission)

    def density(self) -> Density:
        """Return its density."""
        return self._density

    def check_lines(self, lines: tuple[str, ...] | None) -> None:
        if lines is not None and STARS in lines and self.kinematics() is None:
            raise ConfigError(
                "a spectral cube sees the stars of a disk with kinematics "
                "only (its stars have no velocities to place in channels)")

    def kinematics(self) -> DiskTraitsKinematics | None:
        """Return its kinematics, if any."""
        return self._kinematics

    def clouds(self) -> Clouds | None:
        """Return its clouds, if it is made of clouds."""
        return self._clouds

    def thick(self, seen_by: 'type[Emission]') -> bool:
        return self._density.thick()

    def frame_params(self) -> FrameParams:
        return self._disk.frame_params()

    def pdescs(self) -> dict[str, ParamDesc]:
        lines = self._lines.pdescs() if self._lines is not None else {}
        bands = self._bands.pdescs() if self._bands is not None else {}
        return self._disk.pdescs() | lines | bands

    def constants(self) -> dict[str, Any]:
        return dict(
            rnodes=self._disk.rnodes(), subrnodes=self._disk.subrnodes())

    def needs(self) -> tuple[Need, ...]:
        return self._disk.needs(self.name())

    def uses_mass(self) -> bool:
        return self._disk.uses_mass()

    def bounding_radius(self) -> float:
        return self._disk.bounding_radius()

    def density_disk(self) -> _disk.NodalDisk:
        """
        Return its disk: the mass of its galaxy can follow the density of
        the disk (see mass.SurfaceComponent and mass.VerticalComponent).
        """
        return self._disk

    def density_kind(self) -> 'type[DiskDensity]':
        return DiskDensity

    def plan_stage(
            self,
            driver: Driver,
            sky_origin: tuple[float, float] | None,
            followed: bool,
            label: str
    ) -> _stage.DiskStage | None:
        return _stage.plan_disk_stage(
            self._disk, self.emission(), driver, sky_origin, followed, label)

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
