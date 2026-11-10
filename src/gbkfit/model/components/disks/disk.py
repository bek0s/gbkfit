"""
The disk components: a disk of rings with traits, thin (ThinDisk) or
thick (ThickDisk), smooth or made of clouds (a thick one; see Clouds).
"""

import abc
from collections.abc import Sequence
from typing import Any, Self

import numpy as np

from gbkfit.config import check_name, check_options, report_unknown
from gbkfit.driver import Driver
from gbkfit.grid import Grid
from gbkfit.params import ParamDesc
from gbkfit.utils import iterutils
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
from ..parts import KinematicTraits, density_traits, kinematic_traits


__all__ = [
    'Disk',
    'ThickDisk',
    'ThinDisk'
]


# The options of a thick disk that a thin one does not take
_THICK_OPTIONS = (
    'rhtraits', 'vhtraits', 'dhtraits', 'opacity', 'zptraits', 'clouds')


def _fluid(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """
    Return the names of the mass disks of the fluid of a disk as a tuple
    (none if not given). Raise ValueError if a name is not valid or
    repeats.
    """
    names = iterutils.tuplify(value) if value is not None else ()
    if value is not None and not names or len(set(names)) != len(names):
        raise ValueError(
            f"the fluid of a disk names mass disks, at least one and each "
            f"once; it is {list(names)}")
    for name in names:
        check_name(name)
    return names


def _check_balance(
        kinematics: KinematicTraits | None,
        support: traits.PressureSupport | None,
        fluid: tuple[str, ...],
        thick: bool
) -> None:
    """
    Check the vertical balance and the pressure of a disk of the given
    kinematics, pressure support and fluid, thick or not: a vertical
    equilibrium holds up the thickness of a thick disk; the pressure
    corrects the rotation of its velocity trait 'mass', and at each
    height (with the velocity height trait 'mass'), that of
    'self_gravitating' in the sheet of the surface density of its fluid;
    a fluid is what the pressure or the equilibrium 'sheet' takes.

    Raises
    ------
    ValueError
        If one of these does not hold.
    """
    equilibrium = kinematics.equilibrium() if kinematics is not None \
        else None
    if equilibrium is not None and not thick:
        raise ValueError(
            "the vertical equilibrium holds up the thickness of a thick "
            "disk: a thin disk has none")
    vptraits = kinematics.vptraits() if kinematics is not None else ()
    if equilibrium is not None and equilibrium.gravity() == 'rotation' \
            and not any(trait.direction() == 'tan' for trait in vptraits):
        raise ValueError(
            "the vertical equilibrium 'rotation' takes the rotation of the "
            "tangential velocity traits of the disk: it has none")
    if support is not None and not any(
            isinstance(trait, traits.VPTraitMass) for trait in vptraits):
        raise ValueError(
            "the pressure support corrects the rotation of the disk in the "
            "gravity of its galaxy: give it the velocity trait 'mass'")
    vhtraits = (kinematics.vhtraits() or ()) if kinematics is not None \
        else ()
    if support is not None and support.type() == 'self_gravitating' \
            and not fluid and any(
                isinstance(trait, traits.VHTraitMass) for trait in vhtraits):
        raise ValueError(
            "the velocity height trait 'mass' takes the self_gravitating "
            "pressure support at each height in the sheet of the surface "
            "density of its fluid, of the scale height sigma^2 / (pi G "
            "Sigma): give the disk its fluid")
    sheet = equilibrium is not None and equilibrium.gravity() == 'sheet'
    if fluid and support is None and not sheet:
        raise ValueError(
            "the fluid of a disk is the gas whose density its pressure "
            "support and its equilibrium 'sheet' take: it has neither")


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
            self, self._disk, self._rnodes, self._kinematics)
        if self._clouds is not None:
            info['clouds'] = self._clouds.dump()
        return info

    def __init__(
            self,
            rptraits: traits.RPTrait | Sequence[traits.RPTrait],
            rhtraits: traits.RHTrait | Sequence[traits.RHTrait] | None,
            vptraits: traits.VPTrait | Sequence[traits.VPTrait] | None,
            dptraits: traits.DPTrait | Sequence[traits.DPTrait] | None,
            vhtraits:
            traits.VHTrait | Sequence[traits.VHTrait | None] | None,
            dhtraits:
            traits.DHTrait | Sequence[traits.DHTrait | None] | None,
            pressure_support: traits.PressureSupport | None,
            fluid: str | Sequence[str] | None,
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
        thick = self.thick(Emission)
        rptraits, rhtraits = density_traits(rptraits, rhtraits, thick)
        self._kinematics = kinematics = kinematic_traits(
            vptraits, dptraits, vhtraits, dhtraits)
        fluid = _fluid(fluid)
        _check_balance(kinematics, pressure_support, fluid, thick)
        self._clouds = clouds
        # In the form it was given in, for its configuration (see dump)
        self._rnodes = rnodes = as_nodes(rnodes)
        self._disk = _detail.make_disk(
            rptraits, rhtraits, kinematics, pressure_support, fluid,
            emission, opacity, zptraits, sptraits, thick=thick, loose=loose,
            tilted=tilted,
            rnodes=rnodes, rstep=rstep, interp=interp, clouds=clouds)
        self._lines = make_lines(self._disk.pdescs(), emission)
        self._bands = make_bands(self._disk.pdescs(), emission)

    def check_lines(self, lines: tuple[str, ...] | None) -> None:
        if lines is not None and STARS in lines and self.kinematics() is None:
            raise ValueError(
                "a spectral cube sees the stars of a disk with kinematics "
                "only (its stars have no velocities to place in channels)")

    def kinematics(self) -> KinematicTraits | None:
        """Return its kinematic traits, if it moves."""
        return self._kinematics

    def equilibrium(self) -> traits.DPTraitEquilibrium | None:
        """
        Return its vertical equilibrium, if any (a thick disk's: the last
        of its dispersion traits, see traits.DPTraitEquilibrium).
        """
        return self._disk.equilibrium()

    def pressure_support(self) -> traits.PressureSupport | None:
        """Return its pressure support, if any (see traits.VPTraitMass)."""
        return self._disk.pressure_support()

    def fluid(self) -> tuple[str, ...]:
        """
        Return the names of the mass disks of its fluid, if given (see
        traits.PressureSupport and traits.BalanceGravity).
        """
        return self._disk.fluid()

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
            self._disk, self.emission(), driver, sky_origin, followed, label,
            self.name())

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
    rptraits : RPTrait or Sequence of RPTrait
        Its density polar traits: the density per unit area of its plane,
        seen face-on (parameters 'rpt_*'). What the density is depends on
        its roles: light for a disk that emits.
    vptraits : VPTrait or Sequence of VPTrait, optional
        Its velocity polar traits (parameters 'vpt_*'; see
        parts.KinematicTraits). A disk with vptraits and dptraits moves;
        one that emits lines must.
    dptraits : DPTrait or Sequence of DPTrait, optional
        Its velocity dispersion polar traits (parameters 'dpt_*'),
        isotropic or each along a direction: once one is along a
        direction, every direction needs its own or an isotropic one.
    pressure_support : PressureSupport, optional
        How the rotation of its velocity trait 'mass' is corrected for
        its pressure: none (the default), 'jeans' or 'self_gravitating'
        (see traits.PressureSupport).
    fluid : str or Sequence of str, optional
        The names of the mass disks of its galaxy whose densities its
        pressure support takes instead of its own (e.g. its atomic and
        molecular gas), and whose surface density holds a thick disk up
        in the equilibrium 'sheet' (by default, the mass disks that follow
        the disk).
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

    @staticmethod
    def type_title() -> str:
        return 'Thin disk'

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        # The options of a thick disk are reported as such, not as unknown
        # ones (left out in lenient mode, as unknown ones are)
        if given := [name for name in _THICK_OPTIONS if name in info]:
            report_unknown(
                "options of a thick disk (type: thick_disk), which a thin "
                "disk does not take", given, ())
            info = {k: v for k, v in info.items() if k not in given}
        return super().load(info)

    def __init__(
            self,
            rptraits: traits.RPTrait | Sequence[traits.RPTrait],
            vptraits: traits.VPTrait | Sequence[traits.VPTrait] | None = None,
            dptraits: traits.DPTrait | Sequence[traits.DPTrait] | None = None,
            pressure_support: traits.PressureSupport | None = None,
            fluid: str | Sequence[str] | None = None,
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
            rptraits, None, vptraits, dptraits, None, None, pressure_support,
            fluid, emission, None, None, sptraits, loose, tilted, rnodes,
            rstep, interp, None, name)

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
    rptraits : RPTrait or Sequence of RPTrait
        Its density polar traits: the density per unit area of its plane,
        seen face-on (parameters 'rpt_*'). What the density is depends on
        its roles: light for a disk that emits, the optical depth for
        dust.
    rhtraits : RHTrait or Sequence of RHTrait
        Its density height traits, the probability density of the
        distance from its plane: one that every polar trait shares
        (parameters 'rht_*'), or one for each polar trait, in their order
        (parameters 'rht_*', 'rht2_*', ...).
    vptraits, dptraits : optional
        Its velocity and dispersion polar traits (as those of ThinDisk).
        A disk with both moves; one that emits lines must.
    vhtraits : VHTrait or Sequence of VHTrait, optional
        Its velocity height traits, one shared or one for each velocity
        polar trait (as rhtraits; None items or none at all: the velocity
        does not change with the height).
    dhtraits : DHTrait or Sequence of DHTrait, optional
        Its dispersion height traits, one shared or one for each fitted
        dispersion polar trait (as vhtraits; its vertical equilibrium, the
        last of its dispersion traits if any, takes none: see
        traits.DPTraitEquilibrium).
    pressure_support, fluid : optional
        As those of ThinDisk.
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

    @staticmethod
    def type_title() -> str:
        return 'Thick disk'

    def __init__(
            self,
            rptraits: traits.RPTrait | Sequence[traits.RPTrait],
            rhtraits: traits.RHTrait | Sequence[traits.RHTrait],
            vptraits: traits.VPTrait | Sequence[traits.VPTrait] | None = None,
            dptraits: traits.DPTrait | Sequence[traits.DPTrait] | None = None,
            vhtraits:
            traits.VHTrait | Sequence[traits.VHTrait | None] | None = None,
            dhtraits:
            traits.DHTrait | Sequence[traits.DHTrait | None] | None = None,
            pressure_support: traits.PressureSupport | None = None,
            fluid: str | Sequence[str] | None = None,
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
            rptraits, rhtraits, vptraits, dptraits, vhtraits, dhtraits,
            pressure_support, fluid, emission, opacity, zptraits, sptraits,
            loose, tilted, rnodes, rstep, interp, clouds, name)

    def thick(self, seen_by: 'type[Emission]') -> bool:
        return True
