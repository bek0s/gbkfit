"""
The kinematics of a spheroid (SpheroidKinematics): how its stars move,
in the gravity of its galaxy (JeansKinematics, see _jeans) or as its
traits give (TraitsKinematics); and their motion in one spheroid
(SpheroidMotion): their parameters, and their values in its plan.
"""

import abc
from collections.abc import Sequence
from typing import Any

import numpy as np

from gbkfit.driver import Driver
from gbkfit.math.interpolation import Interpolator
from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from .. import traits
from .._native import ParamBuffer
from .._nodal import (
    NodalTraits, TraitTable, native_traits, parse_component_rnode_args,
    subring_radii, with_default_height_traits)
from .._stage import GalaxyMass
from ..base import ComponentState
from ..parts import Kinematics


__all__ = [
    'SPHEROID_TRAITS',
    'SpheroidKinematics',
    'SpheroidLight',
    'SpheroidMotion',
    'TraitsKinematics',
    'TraitsMotion'
]


# The light of a spheroid, as its kinematics take it (see Spheroid.light):
# the terms of its profile, their function and its table, if any
SpheroidLight = tuple[
    np.ndarray, tuple[float, float], tuple[np.ndarray, np.ndarray] | None]


class SpheroidMotion(abc.ABC):
    """
    The motion of the stars of one spheroid (see
    SpheroidKinematics.motion): its parameters, its state in each
    evaluation, and its values in the plan of the spheroid, in groups of
    its buffer (see _native.ParamBuffer) that the native spheroid takes
    (see the native SpheroidArgs).
    """

    @abc.abstractmethod
    def pdescs(self) -> dict[str, ParamDesc]:
        """Return its parameters, by name."""
        pass

    @abc.abstractmethod
    def takes_light(self) -> bool:
        """
        Check whether it takes the light of its spheroid (see state), so
        that its velocities change with it.
        """
        pass

    @abc.abstractmethod
    def state(
            self,
            values: ParamValues,
            light: SpheroidLight,
            mass: GalaxyMass | None
    ) -> ComponentState | None:
        """
        Return its state in an evaluation, if it has one: of the values of
        the parameters of its spheroid, the light of the spheroid (see
        Spheroid.light) and the mass of its galaxy, if any.
        """
        pass

    @abc.abstractmethod
    def sizes(self) -> dict[str, int]:
        """
        Return the number of values of each of its groups of the buffer
        (see write), by name.
        """
        pass

    @abc.abstractmethod
    def native(
            self, driver: Driver, dtype: np.dtype, buffer: ParamBuffer
    ) -> dict[str, Any]:
        """
        Return the arguments of the native spheroid that describe it, with
        the device memory of its groups of the buffer.
        """
        pass

    @abc.abstractmethod
    def write(
            self,
            buffer: ParamBuffer,
            params: ParamValues,
            state: ComponentState | None
    ) -> None:
        """
        Write its values into the host memory of its groups of the buffer,
        at the values of the parameters of its spheroid and its state.
        """
        pass


class SpheroidKinematics(parseutils.TypedSerializable, abc.ABC):
    """
    How the stars of a spheroid move (see Spheroid): each point of it
    with a Gaussian distribution of velocities, of a mean and a dispersion
    on the line of sight, in the frame of a thick disk whose spin axis is
    the axis of the spheroid: the systemic velocity vsys and a rotation
    around the axis, and the velocity ellipsoid of the dispersions along
    the radial, tangential and vertical directions of the cylinder.

    The mean and the dispersion of the velocities of a line of sight (or
    of a bin) are then those of its stars, but not the shape of their
    distribution, which a Gaussian at each point does not give: its h3,
    h4 and the sigma of a Gauss-Hermite series fitted to it (e.g. by pPXF)
    are not those of the stars (a sigma 5% low, in a test of an isotropic
    Hernquist sphere). Fit their mean and dispersion: observables of
    moments of the method 'moments', with the data of pPXF converted (see
    prep.gauss_hermite_moments).
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    @abc.abstractmethod
    def uses_mass(self) -> bool:
        """
        Check whether its stars move in the gravity of their galaxy: it
        needs the mass of the galaxy and its forces.
        """
        pass

    @abc.abstractmethod
    def motion(self, rmax: float) -> SpheroidMotion:
        """
        Return its motion in a spheroid of the given rmax (arcsec, see
        Spheroid). Raise ConfigError if it cannot move so.
        """
        pass


# The kinds of the traits of TraitsKinematics: velocity (v) and dispersion
# (d) polar (p) and height (h) traits, and the prefix of the parameters of
# each
TRAIT_KINDS = ('vpt', 'vht', 'dpt', 'dht')
TRAIT_PREFIXES = dict(vpt='vpt', vht='vht', dpt='dpt', dht='dht')

# The traits that the stars of a spheroid take (see TraitsKinematics): the
# polar traits of the radius and the azimuth alone (as a cone: not the
# mixtures, maps and distortions of the plane of a disk); velocity traits
# around the axis only ('tan': stars do not flow out), and not the
# rotation in the gravity of the galaxy (mass: that is JeansKinematics)
# nor the flows in the wedges, bars and spiral waves of a disk; dispersion
# traits isotropic or along a direction of the cylinder (the velocity
# ellipsoid); and every height trait but the rotation in the gravity at a
# height
SPHEROID_TRAITS = TraitTable('spheroid', dict(
    vpt=tuple(
        cls for cls in traits.vpt_parser.registered_classes().values()
        if cls not in (
            traits.VPTraitMass, traits.VPTraitWedge, traits.VPTraitBar,
            traits.VPTraitNWBar, traits.VPTraitSpiral)),
    vht=tuple(
        cls for cls in traits.vht_parser.registered_classes().values()
        if cls is not traits.VHTraitMass),
    dpt=(
        traits.DPTraitUniform, traits.DPTraitExponential,
        traits.DPTraitGauss, traits.DPTraitGGauss, traits.DPTraitLorentz,
        traits.DPTraitMoffat, traits.DPTraitSech2, traits.DPTraitNWUniform,
        traits.DPTraitNWHarmonic),
    dht=tuple(traits.dht_parser.registered_classes().values())),
    ('tan',), tuple(traits.DPT_DIRECTIONS))


class TraitsKinematics(SpheroidKinematics):
    """
    The kinematics of the stars of a spheroid given by traits (see
    traits), as those of a thick disk whose spin axis is the axis of the
    spheroid: at the distance R from the axis, the azimuth theta and the
    height z along it, the mean velocity is vsys plus the sum of the
    velocity polar traits of (R, theta), speeds around the axis, each
    times its height trait of |z|; the dispersion is that of the velocity
    ellipsoid of the dispersion traits (each polar trait times its height
    trait) on the line of sight: the sum of those along each direction of
    the cylinder and of the isotropic ones (see traits). The height traits
    are of the factor one (the same at every height) unless given. The
    traits are on radial nodes of R, by default 0 and the rmax of the
    spheroid; given ones go from 0 to rmax at least, where its light is.
    Its stars take neither the mass of their galaxy nor its gravity: e.g.
    the kinematics of a spheroid without a model of its mass.

    Parameters
    ----------
    vptraits : VPTrait or Sequence of VPTrait
        The velocity polar traits (parameters 'vpt_*'), around the axis
        ('tan').
    dptraits : DPTrait or Sequence of DPTrait
        The velocity dispersion polar traits (parameters 'dpt_*'),
        isotropic or each along a direction: once one is along a
        direction, every direction needs its own or an isotropic one.
    vhtraits : VHTrait or Sequence of VHTrait, optional
        The velocity height traits, one for each velocity polar trait
        (parameters 'vht_*'; None items or none at all: of the factor one).
    dhtraits : DHTrait or Sequence of DHTrait, optional
        The dispersion height traits, one for each dispersion polar trait
        (parameters 'dht_*'; as vhtraits).
    rnmin, rnmax, rnsep, rnlen, rnodes : optional
        The radii of its nodes, of R (arcsec): rnodes; or rnmin, rnmax and
        rnsep (every rnsep); or rnmin, rnmax and rnlen (rnlen nodes); by
        default 0 and the rmax of the spheroid.
    rstep : float, optional
        The width of the rings it is evaluated on, at most half the
        smallest separation of its radii; by default that, or 1 if less.
    interp : str, optional
        How its node-wise parameters are interpolated between its radii:
        'linear', 'akima' or 'pchip'.

    Raises
    ------
    ConfigError
        If the traits are not valid (see parts.Kinematics), a spheroid
        does not take one of them (see SPHEROID_TRAITS), or the nodes are
        not valid.
    """

    @staticmethod
    def type() -> str:
        return 'traits'

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'TraitsKinematics':
        for parser, key, required in (
                (traits.vpt_parser, 'vptraits', True),
                (traits.dpt_parser, 'dptraits', True),
                (traits.vht_parser, 'vhtraits', False),
                (traits.dht_parser, 'dhtraits', False)):
            parseutils.load_option_and_update_info(
                parser, info, key, required=required)
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        info = super().dump() | self._kinematics.dump()
        if self._nodes is not None:
            info.update(rnodes=list(self._nodes['rnodes']))
        if self._rstep is not None:
            info.update(rstep=self._rstep)
        info.update(interp=self._interp)
        return info

    def __init__(
            self,
            vptraits: traits.VPTrait | Sequence[traits.VPTrait],
            dptraits: traits.DPTrait | Sequence[traits.DPTrait],
            vhtraits:
            traits.VHTrait | Sequence[traits.VHTrait | None] | None = None,
            dhtraits:
            traits.DHTrait | Sequence[traits.DHTrait | None] | None = None,
            rnmin: float | None = None,
            rnmax: float | None = None,
            rnsep: float | None = None,
            rnlen: int | None = None,
            rnodes: Sequence[float] | None = None,
            rstep: float | None = None,
            interp: str = 'linear'
    ):
        self._kinematics = Kinematics(vptraits, dptraits, vhtraits, dhtraits)
        SPHEROID_TRAITS.check(self.traits())
        given = (rnmin, rnmax, rnsep, rnlen, rnodes) != (None,) * 5
        self._nodes = parse_component_rnode_args(
            rnmin, rnmax, rnsep, rnlen, rnodes, rstep, interp) \
            if given else None
        self._rstep = rstep
        self._interp = interp

    def traits(self) -> dict[str, tuple[traits.Trait, ...]]:
        """
        Return its traits by kind (see TRAIT_KINDS), with the height traits
        of the factor one where none are given.
        """
        kinematics = self._kinematics
        return dict(
            vpt=kinematics.vptraits(), dpt=kinematics.dptraits(),
            vht=with_default_height_traits(
                kinematics.vhtraits(), len(kinematics.vptraits()),
                traits.VHTraitOne),
            dht=with_default_height_traits(
                kinematics.dhtraits(), len(kinematics.dptraits()),
                traits.DHTraitOne))

    def uses_mass(self) -> bool:
        return False

    def motion(self, rmax: float) -> 'TraitsMotion':
        nodes = self._nodes if self._nodes is not None \
            else parse_component_rnode_args(
                None, None, None, None, (0.0, rmax), self._rstep,
                self._interp)
        rnodes = nodes['rnodes']
        if rnodes[0] != 0 or rnodes[-1] < rmax:
            raise ConfigError(
                f"the nodes of the traits of a spheroid go from 0 to its "
                f"rmax ({rmax}) at least, where its light is; they go from "
                f"{rnodes[0]} to {rnodes[-1]}")
        return TraitsMotion(self.traits(), **nodes)


class TraitsMotion(SpheroidMotion):
    """
    The motion of the stars of a spheroid of TraitsKinematics: its traits
    on its radial nodes (see NodalTraits), evaluated at its subnodes,
    whose parameters are written to a group of the buffer for each kind.
    """

    def __init__(
            self,
            traits_: dict[str, Sequence[traits.Trait]],
            rnodes: Sequence[float],
            rstep: float,
            interp: type[Interpolator]
    ):
        self._subrnodes = subring_radii(rnodes, rstep)
        self._traits = NodalTraits(
            TRAIT_KINDS, traits_, TRAIT_PREFIXES, rnodes, self._subrnodes,
            interp)

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._traits.pdescs()

    def takes_light(self) -> bool:
        return False

    def state(
            self,
            values: ParamValues,
            light: SpheroidLight,
            mass: GalaxyMass | None
    ) -> None:
        return None

    def sizes(self) -> dict[str, int]:
        return self._traits.sizes()

    def native(
            self, driver: Driver, dtype: np.dtype, buffer: ParamBuffer
    ) -> dict[str, Any]:
        return dict(
            rnodes=driver.mem_copy_h2d(np.array(self._subrnodes, dtype)),
            **native_traits(driver, dtype, self._traits, buffer))

    def write(
            self,
            buffer: ParamBuffer,
            params: ParamValues,
            state: ComponentState | None
    ) -> None:
        # (fill replaces the node-wise values in the parameters it is given)
        self._traits.fill(buffer, dict(params))
