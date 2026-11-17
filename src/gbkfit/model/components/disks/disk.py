"""
The disk components: a disk of rings with traits, thin (ThinDisk) or
thick (ThickDisk), smooth or made of clouds (a thick one; see Clouds).
"""

import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from gbkfit.config import check_name, check_options, report_unknown
from gbkfit.driver import Driver
from gbkfit.params import ParamDesc
from gbkfit.utils import iterutils
from ...mass import DiskDensity
from .. import traits
from .._needs import Need
from .._common import Interp, RNodes, RStep
from .._native import light_pdescs
from .._profiles import ComponentProfiles, traited_families
from ..base import (
    Component, ComponentLinks, ComponentPart, ComponentPlan, FrameParams,
    SceneTarget)
from ..nodes import Nodes, as_nodes
from ..roles import Emission, Opacity, View
from . import _build, _nodal, _profiles, _stage
from .clouds import Clouds
from ..parts import KinematicTraits, density_traits, kinematic_traits

if TYPE_CHECKING:
    from gbkfit.instrument import PSF


__all__ = [
    'Disk',
    'ThickDisk',
    'ThinDisk'
]


# The options of a thick disk that a thin one does not take
_THICK_OPTIONS = ('vhtraits', 'dhtraits', 'opacity', 'zptraits', 'clouds')


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
        height: traits.RHTraitEquilibrium | None,
        support: traits.PressureSupport | None,
        fluid: tuple[str, ...],
        thick: bool
) -> None:
    """
    Check the vertical balance and the pressure of a disk of the given
    kinematics, equilibrium of its thickness (if any), pressure support
    and fluid, thick or not. A disk derives one side of its balance: its
    dispersion (a thick disk's), or its thickness, from its dispersion,
    isothermal in z (which on a thin disk gives its pressure support the
    density at its midplane); the equilibrium 'rotation' takes the
    rotation of its tangential traits; the pressure corrects the rotation
    of its velocity trait 'mass', or without it, of its tangential traits
    (a fitted curve of circular velocity); a fluid is what the pressure or
    an equilibrium 'sheet' takes.

    Raises
    ------
    ValueError
        If one of these does not hold.
    """
    dispersion = kinematics.equilibrium() if kinematics is not None \
        else None
    if dispersion is not None and height is not None:
        raise ValueError(
            "a disk derives one side of its vertical balance: its thickness "
            "(the equilibrium of its rhtraits) or its dispersion (its "
            "option equilibrium), not both")
    if height is not None and kinematics is None:
        raise ValueError(
            "the vertical equilibrium of the thickness of a disk takes its "
            "dispersion: give it vptraits and dptraits")
    if height is not None and any(
            trait is not None and not isinstance(trait, traits.DHTraitOne)
            for trait in kinematics.dhtraits() or ()):
        raise ValueError(
            "the vertical equilibrium of the thickness of a disk takes its "
            "dispersion, isothermal in z: its dhtraits must be 'one'")
    if height is not None and not thick and support is None:
        raise ValueError(
            "a thin disk does not draw its vertical equilibrium, which "
            "gives its pressure support the density at its midplane: give "
            "it pressure_support")
    equilibrium = dispersion if dispersion is not None else height
    vptraits = kinematics.vptraits() if kinematics is not None else ()
    if equilibrium is not None and equilibrium.gravity() == 'rotation' \
            and not any(trait.direction() == 'tan' for trait in vptraits):
        raise ValueError(
            "the vertical equilibrium 'rotation' takes the rotation of the "
            "tangential velocity traits of the disk: it has none")
    if support is not None \
            and not any(trait.direction() == 'tan' for trait in vptraits):
        raise ValueError(
            "the pressure support corrects the rotation of the tangential "
            "velocity traits of the disk: it has none")
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

    config_titles = dict(
        pressure_support='Pressure support', fluid='Gas disks of the mass',
        loose='Centre per ring', tilted='Orientation per ring',
        clouds='Clouds')
    config_references = dict(fluid='mass')

    def dump(self) -> dict[str, Any]:
        info = _build.dump_disk_component(
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
            equilibrium: traits.DispersionEquilibrium | None,
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
        thick = self.has_height()
        rptraits, rhtraits, height = density_traits(
            rptraits, rhtraits, thick)
        self._kinematics = kinematics = kinematic_traits(
            vptraits, dptraits, vhtraits, dhtraits, equilibrium)
        fluid = _fluid(fluid)
        _check_balance(kinematics, height, pressure_support, fluid, thick)
        self._clouds = clouds
        # In the form it was given in, for its configuration (see dump)
        self._rnodes = rnodes = as_nodes(rnodes)
        self._disk = _build.make_disk(
            rptraits, rhtraits, height, kinematics, pressure_support, fluid,
            emission, opacity, zptraits, sptraits, thick=thick, loose=loose,
            tilted=tilted, rnodes=rnodes, rstep=rstep, interp=interp,
            clouds=clouds)
        self._light_pdescs = light_pdescs(self._disk.pdescs(), emission)

    def kinematics(self) -> KinematicTraits | None:
        """Return its kinematic traits, if it moves."""
        return self._kinematics

    def moves(self) -> bool:
        return self._kinematics is not None

    @staticmethod
    @abc.abstractmethod
    def has_height() -> bool:
        """
        Check whether its density has a height (a thick disk), which every
        observation of it needs a depth axis to see.
        """

    def check_psf(self, psf: 'PSF | None') -> None:
        # (the light of its image traits is blurred by their own PSFs)
        for trait in self._disk.traits('rpt'):
            if isinstance(trait, traits.RPTraitImage):
                trait.check_psf(psf)

    def needs_depth(self, view: View) -> bool:
        return self.has_height()

    def rings_gap(self) -> str | None:
        """
        Return why its rings are not ellipses on the sky, as a mass that
        follows it lays them out (see Ringed), if they are not: of vertical
        distortions of an order of 2 or more (those of the orders 0 and 1
        move and tilt a ring, which stays an ellipse).
        """
        orders = sorted({
            trait.order() for trait in self._disk.traits('zpt')
            if trait.order() > 1})
        if not orders:
            return None
        return (
            f"the disk {self.name()!r} has vertical distortions of the "
            f"orders {orders}, whose rings are not ellipses on the sky "
            f"(those of the orders 0 and 1 are)")

    def equilibrium(self) -> traits.DispersionEquilibrium | None:
        """
        Return the vertical equilibrium of its dispersion, if any (a
        thick disk's: the last of its dispersion traits, see
        traits.DispersionEquilibrium).
        """
        return self._disk.equilibrium()

    def height_equilibrium(self) -> traits.RHTraitEquilibrium | None:
        """
        Return the vertical equilibrium of its thickness, if any (its one
        density height trait, see traits.RHTraitEquilibrium).
        """
        return self._disk.height_equilibrium()

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
        return self._disk.pdescs() | self._light_pdescs

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

    def needs(
            self, links: ComponentLinks = ComponentLinks()
    ) -> tuple[Need, ...]:
        # (its vertical balance, against the mass of its galaxy)
        label = f' {self.name()!r}' if self.name() is not None else ''
        gravity = self.balance_gravity()
        if links.mass and gravity == 'rotation':
            raise ValueError(
                f"the vertical equilibrium 'rotation' of the component"
                f"{label} takes the pull of a spherical mass of its "
                f"rotation, which is not that of a galaxy of mass (a disk "
                f"pulls 2.5-3 times harder where it dominates): give it the "
                f"equilibrium 'galaxy' or 'sheet'")
        height = self.height_equilibrium()
        if links.layer and height is not None and height.gravity() == 'galaxy':
            raise ValueError(
                f"a mass disk follows the layer of the component{label}, "
                f"whose thickness is its equilibrium in the force of the "
                f"mass: a mass that takes the thickness that its own force "
                f"sets is not supported yet; give the mass disk a vertical "
                f"structure of its own, or the disk the equilibrium 'sheet'")
        if links.mass and gravity == 'sheet' \
                and not self._disk.galaxy_fluid(links.followers):
            raise ValueError(
                f"the vertical equilibrium 'sheet' of the component{label} "
                f"takes the surface density of its fluid: give it a fluid, "
                f"or a mass disk that follows it")
        return self._disk.needs(self.name(), links.followers)

    def density_disk(self) -> _nodal.NodalDisk:
        """
        Return its disk: the mass of its galaxy can follow the density of
        the disk (see mass.SurfaceComponent and mass.VerticalComponent).
        """
        return self._disk

    def profile_families(self) -> dict[str, tuple[float, float]]:
        return traited_families(self._disk)

    def profiles(self, driver: Driver) -> ComponentProfiles:
        return _profiles.DiskProfiles(
            self._disk, self._disk.plan(driver, 0, np.float64))

    def density_kind(self) -> 'type[DiskDensity]':
        return DiskDensity

    def density_has_height(self) -> bool:
        return self.has_height()

    def balance_gravity(self) -> str | None:
        """
        Return the gravity that its vertical equilibrium balances, of its
        dispersion or its thickness (see traits.BalanceGravity), if it has
        one.
        """
        return self._disk.balance_gravity()

    def plan_stage(
            self,
            driver: Driver,
            sky_origin: tuple[float, float] | None,
            links: ComponentLinks,
            label: str
    ) -> _stage.DiskStage | None:
        return _stage.plan_disk_stage(
            self._disk, self.emission(), driver, sky_origin, links, label)

    def plan(
            self,
            target: SceneTarget,
            emits: bool,
            opacity_ratios: np.ndarray | None,
            redshift: float
    ) -> ComponentPlan:
        return _build.disk_plan(
            self._disk, self.emission(), target.driver, target.spectral,
            target.dtype, target.view, emits, opacity_ratios,
            target.data_unit, redshift)


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
    rhtraits : RHTraitEquilibrium, optional
        Its vertical equilibrium, which it does not draw: the thickness
        that gives its pressure support the density at its midplane (see
        traits.RHTraitEquilibrium); none by default.
    vptraits : VPTrait or Sequence of VPTrait, optional
        Its velocity polar traits (parameters 'vpt_*'; see
        parts.KinematicTraits). A disk with vptraits and dptraits moves;
        one that emits lines must.
    dptraits : DPTrait or Sequence of DPTrait, optional
        Its velocity dispersion polar traits (parameters 'dpt_*'),
        isotropic or each along a direction: once one is along a
        direction, every direction needs its own or an isotropic one.
    pressure_support : PressureSupport, optional
        How the rotation of its velocity trait 'mass', or without it of
        its tangential velocity traits, is corrected for its pressure:
        none (the default) or 'jeans' (see traits.PressureSupport).
    fluid : str or Sequence of str, optional
        The names of the mass disks of its galaxy whose densities its
        pressure support takes instead of its own (e.g. its atomic and
        molecular gas), and whose surface density holds the disk up in
        the equilibrium 'sheet' (by default, the mass disks that follow
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

    config_titles = dict(rhtraits='Vertical equilibrium')
    config_start = dict(
        rnodes=dict(max=16, sep=2), rptraits=dict(type='exponential'),
        vptraits=dict(type='arctan', direction='tan'),
        dptraits=dict(type='uniform'), emission=dict(type='lines'))

    parts = (
        ComponentPart('rnodes'), ComponentPart('rptraits'),
        ComponentPart('vptraits'), ComponentPart('dptraits'),
        ComponentPart('emission'), ComponentPart('sptraits'),
        ComponentPart('pressure_support'), ComponentPart('rhtraits'))

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
            rhtraits: traits.RHTrait | Sequence[traits.RHTrait] | None = None,
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
            rptraits, rhtraits, vptraits, dptraits, None, None, None,
            pressure_support, fluid, emission, None, None, sptraits, loose,
            tilted, rnodes, rstep, interp, None, name)

    @staticmethod
    def has_height() -> bool:
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
        (parameters 'rht_*', 'rht2_*', ...); or its vertical equilibrium,
        the one that they all share, which its galaxy derives (see
        traits.RHTraitEquilibrium).
    vptraits, dptraits : optional
        Its velocity and dispersion polar traits (as those of ThinDisk).
        A disk with both moves; one that emits lines must.
    vhtraits : VHTrait or Sequence of VHTrait, optional
        Its velocity height traits, one shared or one for each velocity
        polar trait (as rhtraits; None items or none at all: the velocity
        does not change with the height).
    dhtraits : DHTrait or Sequence of DHTrait, optional
        Its dispersion height traits, one shared or one for each
        dispersion polar trait (as vhtraits).
    equilibrium : DispersionEquilibrium, optional
        The dispersion of its vertical equilibrium, which its galaxy
        derives and adds in quadrature to that of its dispersion traits
        (see traits.DispersionEquilibrium): with velocity traits, it moves
        with this dispersion alone, or with dispersion traits too.
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

    config_titles = dict(equilibrium='Equilibrium dispersion')
    config_start = ThinDisk.config_start | dict(
        rhtraits=dict(type='sech2', zmax=6))

    parts = (
        ComponentPart('rnodes'), ComponentPart('rptraits', 'rhtraits'),
        ComponentPart('vptraits', 'vhtraits'),
        ComponentPart('dptraits', 'dhtraits'), ComponentPart('equilibrium'),
        ComponentPart('emission'), ComponentPart('zptraits'),
        ComponentPart('sptraits'), ComponentPart('opacity'),
        ComponentPart('clouds'), ComponentPart('pressure_support'))

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
            equilibrium: traits.DispersionEquilibrium | None = None,
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
            equilibrium, pressure_support, fluid, emission, opacity,
            zptraits, sptraits, loose, tilted, rnodes, rstep, interp,
            clouds, name)

    @staticmethod
    def has_height() -> bool:
        return True
