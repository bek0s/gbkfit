"""
The base of the traits (Trait), the bases of their kinds (e.g. RPTrait,
the density polar traits), and what several kinds share.
"""

import abc
import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from gbkfit.params.pdescs import ParamDesc, ParamVectorDesc
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from .options import (
    Blobs, Direction, DispersionDirection, Nodes, SAMPLING_DEFAULT, Sampling,
    TraitOption, VPT_DIRECTIONS, VPT_DIRECTION_PLANE, Zmax, splits_around)

if TYPE_CHECKING:
    from .velocity import PressureSupport


__all__ = [
    'DHTrait',
    'DPTrait',
    'GalaxyInput',
    'HTrait',
    'ImageInput',
    'PTrait',
    'RHTrait',
    'RPTrait',
    'RotationFactorsInput',
    'RotationInput',
    'SPTrait',
    'Trait',
    'VHTrait',
    'VPTrait',
    'VPTraitPlanar',
    'WaveInput',
    'ZPTrait',
    'trait_desc'
]


_Option = TypeVar('_Option', bound=TraitOption)


@dataclasses.dataclass(frozen=True)
class GalaxyInput:
    """
    What a trait takes from its galaxy at each evaluation (see
    Trait.from_galaxy): the values of its parameters params (by their
    names in the trait), which its galaxy computes rather than its
    configuration gives (they are not parameters of its component).
    """
    params: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class RotationInput(GalaxyInput):
    """
    The circular velocity of the mass of the galaxy at the radii where
    the parameters are sampled, corrected for the pressure of their disk
    as support says, if it says (see VPTraitMass).
    """
    support: 'PressureSupport | None'


@dataclasses.dataclass(frozen=True)
class RotationFactorsInput(GalaxyInput):
    """
    The factors of the rotation in the gravity of the galaxy at heights,
    v(R, z) / v(R, 0), at the subnodes: of each parameter, its height
    (arcsec; see VHTraitMass).
    """
    heights: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class WaveInput(GalaxyInput):
    """
    The tables of a density wave in the rotation of the disk, at the
    subnodes (see VPTraitSpiral).
    """


@dataclasses.dataclass(frozen=True)
class ImageInput(GalaxyInput):
    """
    The face-on light of an image in the plane of the disk, which the
    galaxy makes (see RPTraitImage).
    """


class Trait(parseutils.TypedSerializable, abc.ABC):
    """
    A trait of a component made of nodes: a profile of one of its
    properties (see the package docstring). A trait declares its type
    (its name in configurations), its uid (its identifier in the native
    kernels), its parameters (params_sm, params_rnw) and its constants
    (consts), and holds its options (see TraitOption).

    A trait can have a name, which prefixes its parameters in its
    component in place of its kind and position (e.g. 'rot_vt' in place
    of 'vpt1_vt'), so that they keep their names when other traits are
    added; the traits of a kind of a component have names all or none,
    and two traits of a component cannot share one.

    Parameters
    ----------
    *options : TraitOption
        Its options (given by its class).
    name : str, optional
        The name: letters, digits and underscores.

    Raises
    ------
    ConfigError
        If the name is invalid.
    """

    @staticmethod
    @abc.abstractmethod
    def uid() -> int:
        """Return the identifier of the trait in the native kernels."""

    def dump(self) -> dict[str, Any]:
        info: dict[str, Any] = dict(type=self.type())
        if self._name is not None:
            info['name'] = self._name
        for option in self._options:
            info |= option.dump()
        return info

    def __init__(self, *options: TraitOption, name: str | None = None):
        parseutils.check_name(name)
        self._options = options
        self._name = name

    def name(self) -> str | None:
        """Return its name, if any."""
        return self._name

    def _option(self, kind: type[_Option]) -> _Option | None:
        """Return its option of a kind, if it holds one."""
        return next(
            (option for option in self._options if isinstance(option, kind)),
            None)

    def consts(self) -> tuple[float, ...]:
        """
        Return the constants of the trait, in the order the kernels read
        them: those of its options (e.g. the order of a harmonic), here.
        """
        return tuple(c for option in self._options for c in option.consts())

    def params_sm(self) -> dict[str, ParamDesc]:
        """Return its parameters that are not node-wise; none here."""
        return {}

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        """
        Return its node-wise parameters, of nnodes values each; none
        here.
        """
        return {}

    def sampling(self) -> str:
        """
        Return where the values of the node-wise parameters are given (its
        option sampling, see Sampling).
        """
        sampling = self._option(Sampling)
        return SAMPLING_DEFAULT if sampling is None else sampling.name

    def from_galaxy(self) -> GalaxyInput | None:
        """
        Return what it takes from its galaxy at each evaluation, if
        anything (see GalaxyInput); nothing here.
        """
        return None


class PTrait(Trait, abc.ABC):
    """
    A polar trait: a function of the radius and the azimuth on the
    surface of its component.
    """

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        """
        Return the azimuths (degrees) at which the azimuthal means of the
        gravity of its galaxy split the circles of the radii r (arcsec, an
        array), so that the trait is smooth between two of them (see
        _profiles): where it jumps, and around its narrow structure (e.g.
        blobs). An array of shape (len(r), n), NaN where a circle has
        fewer than n; None if the trait is smooth on every circle (here).
        params has the values of its parameters, the node-wise ones at the
        subnodes subrnodes, between which the disk interpolates them
        linearly. Here: around its blobs, if any (see Blobs).
        """
        blobs = self._option(Blobs)
        return None if blobs is None else blobs.azimuthal_edges(r, params)


class HTrait(Trait, abc.ABC):
    """
    A height trait: a function of the height above the surface of its
    component, whose parameters can have a value for each node (its
    option rnodes, see Nodes).

    Raises
    ------
    ConfigError
        If the name is invalid, or it has a sampling without rnodes.
    """

    def __init__(self, *options: TraitOption, name: str | None = None):
        super().__init__(*options, name=name)
        if not self.rnodes() and self.sampling() != SAMPLING_DEFAULT:
            raise ConfigError(
                f"sampling is '{self.sampling()}', but rnodes is False: "
                f"the trait has no node-wise parameters")

    def rnodes(self) -> bool:
        """Return whether its parameters are node-wise."""
        nodes = self._option(Nodes)
        return nodes is not None and nodes.rnodes


class RPTrait(PTrait, abc.ABC):
    """
    A density polar trait: the density per unit area of the surface of
    its component (of a disk, seen face-on): light, or the optical depth
    of dust. The Monte Carlo disks draw their
    clouds from it (see has_analytical_integral and cloud_flux).
    """

    @abc.abstractmethod
    def has_analytical_integral(self) -> bool:
        """
        Check whether the Monte Carlo disk makes the clouds of the trait for
        the whole disk at once (True), or for each ring.
        """

    @abc.abstractmethod
    def cloud_flux(
            self, params: dict[str, np.ndarray], subrnodes: np.ndarray
    ) -> np.ndarray:
        """
        Return the flux the Monte Carlo disk shares among the clouds of
        the trait: for the whole disk (one value, with an analytical
        integral) or for each of its rings (around its subnodes subrnodes
        but the first and the last, its edges; params has the values of
        the node-wise parameters at all of them). It is the integral of |B|
        (B is the trait) over the disk or the ring, of the sign of its
        amplitude where that is one number; else the kernel gives each
        cloud the sign of B where it draws it (e.g. a harmonic).
        """

    @abc.abstractmethod
    def axisymmetric(self) -> bool:
        """
        Check whether its value is the same at every azimuth (at each
        radius): then the azimuthal means of it that the gravity of its
        galaxy takes are its value at one azimuth (see NodalDiskDensity and
        pressure_support).
        """

    def amplitudes(self) -> tuple[str, ...]:
        """
        Return the names of its parameters in the unit of its density
        (e.g. its amplitude): their descriptions have no unit, which its
        component sets (e.g. the unit of its light, see NodalTraits).
        Here, a.
        """
        return ('a',)


class RHTrait(HTrait, abc.ABC):
    """
    A density height trait: the probability density of the height of
    the density above the surface (the midplane of a disk), which
    integrates to 1.

    The density lies within zmax of the surface (its option zmax, see
    Zmax), which vertical distortion traits displace: the profile is
    truncated there and renormalised, so that each column keeps its
    density (none where less than 0.001 of the profile is within zmax).
    With the zmax of the vertical distortion traits, it bounds a disk
    along z, so that the depth axis of an observation can hold all of it
    (see Depth).
    """

    def zmax(self) -> float:
        """Return its zmax (arcsec): a bound along z."""
        return self._option(Zmax).value

    def consts(self) -> tuple[float, ...]:
        return (self.zmax(), self.rnodes())

    @abc.abstractmethod
    def cdf(self, z: np.ndarray, *params: np.ndarray) -> np.ndarray:
        """
        Return the cumulative distribution of the profile, not truncated,
        at the heights z above the midplane, for the values of its
        parameters (s, or s and b), broadcast together.
        """


class VPTrait(PTrait, abc.ABC):
    """
    A velocity polar trait: a speed along a direction (its option
    direction, see Direction), projected on the line of sight (see the
    package docstring).
    """

    def direction(self) -> str:
        """Return the direction of its speed (see VPT_DIRECTIONS)."""
        return self._option(Direction).name

    def direction_uid(self) -> int:
        """Return the identifier of its direction in the native kernels."""
        return VPT_DIRECTIONS[self.direction()].uid

    def speed(self) -> str:
        """Return the name of its speed parameter, by its direction."""
        return VPT_DIRECTIONS[self.direction()].speed

    def axisymmetric(self) -> bool:
        """
        Check whether its speed is the same at every azimuth (at each
        radius): then the azimuthal means of it that the vertical
        equilibrium of its disk takes are its value at one azimuth (see
        parts.Equilibrium); here, a speed of the radius only.
        """
        return True


class VPTraitPlanar(VPTrait, abc.ABC):
    """
    A planar velocity polar trait: a velocity in the plane of a disk, with
    a radial and a tangential part (see the package docstring). Its kernel
    gives v_y, its part along the azimuth 90 degrees, which is projected
    on the line of sight as v_y sin(incl).
    """

    def direction(self) -> str:
        return 'plane'

    def direction_uid(self) -> int:
        return VPT_DIRECTION_PLANE

    def axisymmetric(self) -> bool:
        return False


class VHTrait(HTrait, abc.ABC):
    """
    A velocity height trait: a factor of the height, which multiplies its
    velocity polar trait.
    """

    def consts(self) -> tuple[float, ...]:
        return (self.rnodes(),)


class DPTrait(PTrait, abc.ABC):
    """
    A velocity dispersion polar trait: isotropic, or along one direction
    of a disk (its option direction, see DispersionDirection and the
    package docstring).
    """

    def direction(self) -> str:
        """Return the direction of its dispersion (see DPT_DIRECTIONS)."""
        return self._option(DispersionDirection).name

    def direction_uid(self) -> int:
        """Return the identifier of its direction in the native kernels."""
        return self._option(DispersionDirection).uid()

    @abc.abstractmethod
    def axisymmetric(self) -> bool:
        """
        Check whether its value is the same at every azimuth (at each
        radius): then the azimuthal means of it that pressure support
        takes are its value at one azimuth (see pressure_support).
        """


class DHTrait(HTrait, abc.ABC):
    """
    A velocity dispersion height trait: a factor of the height, which
    multiplies its dispersion polar trait.
    """

    def consts(self) -> tuple[float, ...]:
        return (self.rnodes(),)


class ZPTrait(PTrait, abc.ABC):
    """
    A vertical distortion polar trait: a displacement of the plane of the
    disk (its midplane) along z, within zmax of the plane of the galaxy
    (its option zmax, see Zmax), the valid range of its amplitude a. A
    larger one (e.g. of an interpolation of a between the rings that
    overshoots) is limited to zmax; with extra outputs, a warning says
    where. With the zmax of the density height traits, it bounds the disk
    along z, so that the depth axis of an observation can hold all of it
    (see Depth).
    """

    def zmax(self) -> float:
        """Return its zmax (arcsec): a bound along z."""
        return self._option(Zmax).value

    def _amplitude_desc(self, nnodes: int) -> ParamVectorDesc:
        """Return the description of its node-wise amplitude, a."""
        return ParamVectorDesc(
            nnodes, desc="the amplitude of the displacement",
            minimum=-self.zmax(), maximum=self.zmax(), unit='arcsec')


class SPTrait(PTrait, abc.ABC):
    """
    A selection polar trait: 1 where it selects its component, 0
    elsewhere.

    What it selects has sharp edges, which its parameters place: the
    model, which samples its component at points, does not change while
    an edge moves between two of them, so that fits by the slopes of the
    chi-square (least squares) do not move its parameters, and give them
    errors of 0. Fit them with a fitter that takes no slopes (e.g.
    CMA-ES, or a sampler).
    """

    @abc.abstractmethod
    def axisymmetric(self) -> bool:
        """
        Check whether its value is the same at every azimuth (at each
        radius): then the azimuthal means of the density it selects, which
        the gravity of its galaxy takes, can be its value at one azimuth
        (see NodalDiskDensity and pressure_support).
        """


def trait_desc(cls: type[Trait]) -> str:
    """Describe a class of traits, for messages (with its kind)."""
    descs = {
        RPTrait: 'density polar trait',
        RHTrait: 'density height trait',
        VPTrait: 'velocity polar trait',
        VHTrait: 'velocity height trait',
        DPTrait: 'velocity dispersion polar trait',
        DHTrait: 'velocity dispersion height trait',
        ZPTrait: 'vertical distortion polar trait',
        SPTrait: 'selection polar trait'}
    label = next(v for k, v in descs.items() if issubclass(cls, k))
    return parseutils.make_typed_desc(cls, label)


def params_nw_distortion(
        nnodes: int, unit: str | None) -> dict[str, ParamDesc]:
    """
    Return the node-wise parameters of a distortion (see
    RPTraitNWDistortion): its amplitude a, of the given unit (None for a
    density, whose component knows it), azimuth p and width s.
    """
    return dict(
        a=ParamVectorDesc(nnodes, unit=unit),
        p=ParamVectorDesc(nnodes, unit='deg'),
        s=ParamVectorDesc(nnodes, unit='arcsec'))


def splits_nw_distortion(
        r: np.ndarray,
        params: dict[str, np.ndarray],
        subrnodes: Sequence[float]
) -> np.ndarray:
    """
    Return the azimuths (degrees) at which the azimuthal means split the
    circles of the radii r around a distortion (see RPTraitNWDistortion):
    around its azimuth p, by its width s, and opposite (where its
    distance to p turns).
    """
    p = np.interp(r, subrnodes, params['p'])[:, None]
    s = np.interp(r, subrnodes, params['s'])[:, None]
    return np.concatenate(
        (splits_around(r, p, s, np.zeros_like(s)), p + 180), axis=1)
