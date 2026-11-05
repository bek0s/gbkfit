"""
The disk components: a disk of rings with traits, thin (ThinDisk) or
thick (ThickDisk), smooth or made of clouds (a thick one; see Clouds).
"""

import abc
from collections.abc import Mapping, Sequence
from typing import Any, Self

import numpy as np

from gbkfit.config import check_options, config_path, report_unknown
from gbkfit.driver import Driver
from gbkfit.grid import Grid
from gbkfit.params import ParamDesc
from ...mass import DiskDensity
from .. import traits
from .._needs import Need
from .._detail import Interp, RNodes, RStep
from .._native import make_bands, make_lines
from .._units import DataUnit
from ..base import Component, ComponentPlan, FrameParams
from ..lines import Band
from ..nodes import Nodes, as_nodes
from ..roles import STARS, Emission, Opacity
from . import _detail, _disk, _stage
from .clouds import Clouds
from ..parts import (
    Density, ThickDiskTraitsKinematics, ThinDensity,
    ThinDiskTraitsKinematics)


__all__ = [
    'Disk',
    'ThickDisk',
    'ThinDisk'
]


# The options of a thick disk that a thin one does not take, by the
# option of the disk they are in ('': the disk itself)
_THICK_OPTIONS = {
    '': ('opacity', 'zptraits', 'clouds'),
    'density': ('htraits',),
    'kinematics': ('vhtraits', 'dhtraits', 'equilibrium')}


class Disk(Component, abc.ABC):
    """
    A disk of rings, whose properties are the traits of its parts (see
    traits): the base of the thin disks (ThinDisk), whose light is in
    their plane, and the thick ones (ThickDisk), whose density has a
    height. Its rings are given by their radii (rnodes), where its
    node-wise parameters are given; beyond its last ring, its density goes
    on as an exponential of the density and its slope there (see
    traits.RPTrait.tail_scale), and its other node-wise parameters keep
    their values there.
    """

    def dump(self) -> dict[str, Any]:
        info = _detail.dump_disk_component(
            self, self._disk, self._rnodes, self._density, self._kinematics)
        if self._clouds is not None:
            info['clouds'] = self._clouds.dump()
        return info

    def __init__(
            self,
            density: ThinDensity | Density,
            kinematics:
            ThinDiskTraitsKinematics | ThickDiskTraitsKinematics | None,
            emission: Emission | None,
            opacity: Opacity | None,
            zptraits: traits.ZPTrait | Sequence[traits.ZPTrait] | None,
            sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None,
            loose: bool,
            tilted: bool,
            rnodes: Nodes | RNodes | None,
            rstep: RStep | None,
            interp: Interp,
            clouds: Clouds | None,
            name: str | None
    ):
        # (the options are checked by the constructor of the subclass)
        super().__init__(name, emission, opacity)
        self._density = density
        self._kinematics = kinematics
        self._clouds = clouds
        # In the form it was given in, for its configuration (see dump)
        self._rnodes = rnodes = as_nodes(rnodes)
        self._disk = _detail.make_disk(
            density, kinematics, emission, opacity, zptraits, sptraits,
            loose=loose, tilted=tilted, rnodes=rnodes, rstep=rstep,
            interp=interp,
            clouds=clouds)
        self._lines = make_lines(self._disk.pdescs(), emission)
        self._bands = make_bands(self._disk.pdescs(), emission)

    def density(self) -> ThinDensity | Density:
        """Return its density."""
        return self._density

    def check_lines(self, lines: tuple[str, ...] | None) -> None:
        if lines is not None and STARS in lines and self.kinematics() is None:
            raise ValueError(
                "a spectral cube sees the stars of a disk with kinematics "
                "only (its stars have no velocities to place in channels)")

    def kinematics(
            self
    ) -> ThinDiskTraitsKinematics | ThickDiskTraitsKinematics | None:
        """Return its kinematics, if any."""
        return self._kinematics

    def clouds(self) -> Clouds | None:
        """Return its clouds, if it is made of clouds."""
        return self._clouds

    def frame_params(self) -> FrameParams:
        return self._disk.frame_params()

    def pdescs(self) -> dict[str, ParamDesc]:
        lines = self._lines.pdescs() if self._lines is not None else {}
        bands = self._bands.pdescs() if self._bands is not None else {}
        return self._disk.pdescs() | lines | bands

    def param_owners(self) -> dict[str, tuple[str | int, ...]]:
        owners = super().param_owners()
        owners.update({
            name: option
            for name, option in self._disk.trait_owners().items()
            if name in owners})
        return owners

    def constants(self) -> dict[str, Any]:
        return dict(
            rnodes=self._disk.rnodes(), subrnodes=self._disk.subrnodes())

    def needs(self) -> tuple[Need, ...]:
        return self._disk.needs(self.name())

    def uses_mass(self) -> bool:
        return self._disk.uses_mass()

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
            spectral: Grid,
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


class ThinDisk(Disk):
    """
    A disk of rings whose light is in its plane. It is evaluated at every
    pixel of the grid (where its plane crosses the line of sight of the
    pixel, on a grid with a depth axis), and its properties are the traits
    of its parts (see traits). A disk whose density has a height, with
    opacity, vertical distortions or clouds, is a thick disk (see
    ThickDisk).

    Parameters
    ----------
    density : ThinDensity
        Its density.
    kinematics : ThinDiskTraitsKinematics, optional
        How it moves; required to emit lines.
    emission : Emission, optional
        The light it emits.
    sptraits : SPTrait or Sequence of SPTrait, optional
        Its selection polar traits: the parts of the disk that are kept
        (e.g. a range of azimuths), for all its roles.
    loose : bool, optional
        Whether its centre and systemic velocity are node-wise (a value
        for each ring).
    tilted : bool, optional
        Whether its position angle and inclination are node-wise.
    rnodes : Nodes or Sequence of float
        The radii of its rings (arcsec), from 0: given, or evenly spaced
        (in configurations {max, sep} or {max, len}; see Nodes), which its
        configuration keeps. They say where its node-wise parameters
        are given, not where it ends: beyond its last radius, its density
        goes on as an exponential of the density and its slope there (see
        traits.RPTrait.tail_scale), and its other node-wise parameters
        keep their values there.
    rstep : float, optional
        The width of the rings it is evaluated on, at most half the
        smallest separation of its radii; by default that, or 1 if less.
    interp : str, optional
        How its node-wise parameters are interpolated between its radii:
        'linear', 'akima' or 'pchip'.
    name : str, optional
        Its name (see Component).

    Raises
    ------
    ValueError
        If the options are not valid (e.g. the rings), or its parts and
        roles do not go together (e.g. lines without kinematics).
    """

    @staticmethod
    def type() -> str:
        return 'thin_disk'

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        # The options of a thick disk are reported as such, not as unknown
        # ones (left out in lenient mode, as unknown ones are)
        info = dict(info)
        for option, names in _THICK_OPTIONS.items():
            part = info.get(option, {}) if option else info
            if not isinstance(part, Mapping) \
                    or not (given := [n for n in names if n in part]):
                continue
            with config_path(*([option] if option else [])):
                report_unknown(
                    "options of a thick disk (type: thick_disk), which a "
                    "thin disk does not take", given, ())
            kept = {k: v for k, v in part.items() if k not in given}
            info = info | {option: kept} if option else kept
        return super().load(info)

    def __init__(
            self,
            density: ThinDensity,
            kinematics: ThinDiskTraitsKinematics | None = None,
            emission: Emission | None = None,
            sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None = None,
            loose: bool = False,
            tilted: bool = False,
            rnodes: Nodes | RNodes | None = None,
            rstep: RStep | None = None,
            interp: Interp = 'linear',
            name: str | None = None
    ):
        check_options(ThinDisk.__init__, locals())
        super().__init__(
            density, kinematics, emission, None, None, sptraits, loose,
            tilted, rnodes, rstep, interp, None, name)

    def density(self) -> ThinDensity:
        """Return its density."""
        return self._density

    def kinematics(self) -> ThinDiskTraitsKinematics | None:
        """Return its kinematics, if any."""
        return self._kinematics

    def thick(self, seen_by: 'type[Emission]') -> bool:
        return False


class ThickDisk(Disk):
    """
    A disk of rings whose density has a height: smooth, or made of
    clouds. It is evaluated at every voxel of a grid with a depth axis
    (its clouds drawn at random from its density; see Clouds), and its
    properties are the traits of its parts (see traits), with the height
    of each density trait (its density height traits).

    Parameters
    ----------
    density : Density
        Its density, with height traits.
    kinematics : ThickDiskTraitsKinematics, optional
        How it moves (with its vertical equilibrium, if any); required to
        emit lines.
    emission : Emission, optional
        The light it emits.
    opacity : Opacity, optional
        What it removes from the light behind it.
    zptraits : ZPTrait or Sequence of ZPTrait, optional
        Its vertical distortion polar traits (e.g. a warp out of its
        plane).
    sptraits : SPTrait or Sequence of SPTrait, optional
        Its selection polar traits: the parts of the disk that are kept
        (e.g. a range of azimuths), for all its roles.
    loose : bool, optional
        Whether its centre and systemic velocity are node-wise (a value
        for each ring).
    tilted : bool, optional
        Whether its position angle and inclination are node-wise.
    rnodes : Nodes or Sequence of float
        The radii of its rings (arcsec), from 0: given, or evenly spaced
        (in configurations {max, sep} or {max, len}; see Nodes), which its
        configuration keeps. They say where its node-wise parameters
        are given, not where it ends: beyond its last radius, its density
        goes on as an exponential of the density and its slope there (see
        traits.RPTrait.tail_scale), and its other node-wise parameters
        keep their values there.
    rstep : float, optional
        The width of the rings it is evaluated on, at most half the
        smallest separation of its radii; by default that, or 1 if less.
    interp : str, optional
        How its node-wise parameters are interpolated between its radii:
        'linear', 'akima' or 'pchip'.
    clouds : Clouds, optional
        Its clouds, if it is made of clouds; by default, it is smooth.
    name : str, optional
        Its name (see Component).

    Raises
    ------
    ValueError
        If the options are not valid (e.g. the rings), or its parts and
        roles do not go together (e.g. lines without kinematics).
    """

    @staticmethod
    def type() -> str:
        return 'thick_disk'

    def __init__(
            self,
            density: Density,
            kinematics: ThickDiskTraitsKinematics | None = None,
            emission: Emission | None = None,
            opacity: Opacity | None = None,
            zptraits: traits.ZPTrait | Sequence[traits.ZPTrait] | None = None,
            sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None = None,
            loose: bool = False,
            tilted: bool = False,
            rnodes: Nodes | RNodes | None = None,
            rstep: RStep | None = None,
            interp: Interp = 'linear',
            clouds: Clouds | None = None,
            name: str | None = None
    ):
        check_options(ThickDisk.__init__, locals())
        super().__init__(
            density, kinematics, emission, opacity, zptraits, sptraits,
            loose, tilted, rnodes, rstep, interp, clouds, name)

    def density(self) -> Density:
        """Return its density."""
        return self._density

    def kinematics(self) -> ThickDiskTraitsKinematics | None:
        """Return its kinematics, if any."""
        return self._kinematics

    def thick(self, seen_by: 'type[Emission]') -> bool:
        return True
