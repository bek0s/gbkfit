"""
The motion of the stars of a spheroid (SpheroidMotion): its parameters,
what it needs from the galaxy of the spheroid, its state in each
evaluation, and its values in the plan of the spheroid; that of its
traits (TraitsMotion), of the Jeans equations (see _jeans) and of orbits
(see _schwarzschild).
"""

import abc
from collections.abc import Sequence
from typing import Any

import numpy as np

from gbkfit.driver import Driver
from gbkfit.params import ParamDesc, ParamValues
from .. import traits
from ..traits import Trait
from .._native import ParamBuffer
from .._needs import Need
from .._common import Interp, RStep
from .._nodal import (
    NodalTraits, TraitTable, native_traits, parse_component_rnode_args,
    subring_radii, with_default_height_traits)
from ..base import GalaxyMass
from ..base import ComponentState
from ..nodes import Nodes
from ..parts import KinematicTraits


__all__ = [
    'SPHEROID_TRAITS',
    'SpheroidLight',
    'SpheroidMotion',
    'TraitsMotion'
]


# The light of a spheroid, as its motion takes it (see Spheroid.light):
# the terms of its profile, their function and its table, if any
SpheroidLight = tuple[
    np.ndarray, tuple[float, float], tuple[np.ndarray, np.ndarray] | None]


class SpheroidMotion(abc.ABC):
    """
    How the stars of one spheroid move (see Spheroid): its parameters,
    what it needs from the galaxy of the spheroid, its state in each
    evaluation, and its values in the plan of the spheroid, in groups of
    its buffer (see _native.ParamBuffer) that the native spheroid takes
    (see the native SpheroidArgs).
    """

    @staticmethod
    @abc.abstractmethod
    def kind() -> str:
        """
        Return the kind of its kinematics, as messages name them (e.g.
        'jeans': "the jeans kinematics of the spheroid 'stars'").
        """
        pass

    @abc.abstractmethod
    def pdescs(self) -> dict[str, ParamDesc]:
        """Return its parameters, by name."""
        pass

    def needs(self, label: str) -> tuple[Need, ...]:
        """
        Return what it needs from the galaxy of its spheroid (see
        Component.needs), of the given label of its kinematics (e.g. "the
        jeans kinematics of the spheroid 'stars'"); nothing by default.
        """
        return ()

    @abc.abstractmethod
    def takes_light(self) -> bool:
        """
        Check whether it takes the light of its spheroid (see state), so
        that its velocities change with it.
        """
        pass

    def orbit_weights(self) -> str | None:
        """
        Return its parameter of the weights of the orbits of the stars, if
        they move on orbits whose weights are solved (see
        Component.orbit_weights); None by default.
        """
        return None

    def state_params(self) -> tuple[str, ...]:
        """
        Return the names of its parameters that its state reads (see
        state); none by default.
        """
        return ()

    def groundwork(
            self, light: SpheroidLight, mass: GalaxyMass | None
    ) -> object:
        """
        Return the groundwork of its state in an evaluation (see state):
        what the light of its spheroid (see Spheroid.light) and the mass of
        its galaxy, if any, alone set of it; none by default.
        """
        return None

    @abc.abstractmethod
    def state(
            self, values: ParamValues, groundwork: object
    ) -> ComponentState | None:
        """
        Return its state in an evaluation, if it has one: of the values of
        the parameters of its spheroid, those it reads among them (see
        state_params), and its groundwork (see groundwork).
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


# The kinds of the traits of TraitsMotion: velocity (v) and dispersion
# (d) polar (p) and height (h) traits, which name their parameters (see
# _nodal.trait_prefixes)
TRAIT_KINDS = ('vpt', 'vht', 'dpt', 'dht')

# The traits that the stars of a spheroid take (see TraitsMotion): the
# polar traits of the radius and the azimuth alone (as a cone: not the
# mixtures, maps and distortions of the plane of a disk); velocity traits
# around the axis only ('tan': stars do not flow out), and not the
# rotation in the gravity of the galaxy (mass: that is JeansMotion) nor
# the flows in the wedges, bars and spiral waves of a disk; dispersion
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


class TraitsMotion(SpheroidMotion):
    """
    The stars of a spheroid moving as traits give, as those of a thick
    disk whose spin axis is the axis of the spheroid (see TraitSpheroid).
    At the distance R from the axis, the azimuth theta and the height z
    along it, the mean velocity is vsys plus the sum of the velocity polar
    traits of (R, theta), speeds around the axis, each times its height
    trait of |z|; the dispersion is that of the velocity ellipsoid of the
    dispersion traits (each polar trait times its height trait) on the
    line of sight: the sum of those along each direction of the cylinder
    and of the isotropic ones (see traits). The height traits are of the
    factor one (the same at every height) unless given. The traits are on
    radial nodes of R (see NodalTraits), by default 0 and the rmax of the
    spheroid, evaluated at their subnodes, whose parameters are written to
    a group of the buffer for each kind. Its stars take neither the mass
    of their galaxy nor its gravity.

    Parameters
    ----------
    kinematics : KinematicTraits
        The traits (see parts.kinematic_traits).
    rmax : float
        The rmax of the spheroid (arcsec, see Spheroid).
    rnodes : Nodes, optional
        The radii of the nodes (arcsec), from 0 to rmax at least, where
        the light of the spheroid is; by default 0 and rmax.
    rstep : float, optional
        The width of the rings the traits are evaluated on (see
        _nodal.parse_component_rnode_args).
    interp : str, optional
        How the node-wise parameters are interpolated between the radii:
        'linear', 'akima' or 'pchip'.

    Raises
    ------
    ValueError
        If a spheroid does not take one of the traits (see
        SPHEROID_TRAITS), or the nodes are not valid or stop short of
        rmax.
    """

    @staticmethod
    def kind() -> str:
        return 'traits'

    def __init__(
            self,
            kinematics: KinematicTraits,
            rmax: float,
            rnodes: Nodes | None = None,
            rstep: RStep | None = None,
            interp: Interp = 'linear'
    ):
        traits_: dict[str, Sequence[Trait]] = dict(
            vpt=kinematics.vptraits(), dpt=kinematics.dptraits(),
            vht=with_default_height_traits(
                kinematics.vhtraits(), traits.VHTraitOne),
            dht=with_default_height_traits(
                kinematics.dhtraits(), traits.DHTraitOne))
        SPHEROID_TRAITS.check(traits_)
        nodes = parse_component_rnode_args(
            Nodes((0.0, rmax)) if rnodes is None else rnodes, rstep, interp)
        radii = nodes['rnodes']
        if radii[-1] < rmax:
            raise ValueError(
                f"the nodes of the traits of a spheroid go to its rmax "
                f"({rmax}) at least, where its light is; they go to "
                f"{radii[-1]}")
        self._subrnodes = subring_radii(radii, nodes['rstep'])
        # (no density traits: no units of densities)
        self._traits = NodalTraits(
            TRAIT_KINDS, traits_, {}, {}, radii, self._subrnodes,
            nodes['interp'])

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._traits.pdescs()

    def trait_owners(self) -> dict[str, tuple[str | int, ...]]:
        """
        Return the option of each parameter of its traits (see
        NodalTraits.owners), by name.
        """
        return self._traits.owners()

    def traits(self, kind: str) -> tuple[Trait, ...]:
        """Return its traits of one kind (see TRAIT_KINDS)."""
        return self._traits.traits(kind)

    def trait_values(
            self, kind: str, params: ParamValues
    ) -> tuple[dict[str, np.ndarray], ...]:
        """
        Return the values of the parameters of each trait of one kind, by
        their names in the trait, from the values of the parameters of the
        spheroid: the node-wise ones at the subnodes, as it is evaluated.
        """
        return self._traits.values(kind, params)

    def subrnodes(self) -> tuple[float, ...]:
        """Return the radii of its subnodes (arcsec, of R)."""
        return tuple(self._subrnodes)

    def takes_light(self) -> bool:
        return False

    def state(self, values: ParamValues, groundwork: object) -> None:
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
