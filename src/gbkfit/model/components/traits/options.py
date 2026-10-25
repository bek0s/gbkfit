"""
The options of the traits: small values that check themselves and give
their configuration (see TraitOption), which the traits hold (see
Trait). A configuration gives them as flat options of its trait (e.g.
direction: tan, order: 1, sampling: subrings).
"""

import abc
import dataclasses
import numbers
from collections.abc import Sequence
from typing import Any

import numpy as np

from gbkfit.params.pdescs import (
    ParamColumn, ParamDesc, ParamScalarDesc, ParamTable, ParamVectorDesc)
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Blobs',
    'DPT_DIRECTIONS',
    'DPT_DIRECTION_DEFAULT',
    'Direction',
    'DispersionDirection',
    'Harmonic',
    'Nodes',
    'SAMPLINGS',
    'SAMPLING_DEFAULT',
    'Sampling',
    'Spiral',
    'TraitOption',
    'VPTDirection',
    'VPT_DIRECTIONS',
    'Zmax'
]


# The directions of the velocity polar traits: their uids in the native
# kernels
VPT_DIRECTION_TAN = 1
VPT_DIRECTION_RAD = 2
VPT_DIRECTION_VER = 3
VPT_DIRECTION_LOS = 4
VPT_DIRECTION_OUT = 5
VPT_DIRECTION_PLANE = 6
VPT_DIRECTION_AWAY = 7

# The directions of the velocity dispersion polar traits: their uids in
# the native kernels
DPT_DIRECTION_ISO = 0
DPT_DIRECTION_RAD = 1
DPT_DIRECTION_TAN = 2
DPT_DIRECTION_VER = 3

# Where the values of the node-wise parameters of a trait are given: at
# the radial nodes of its disk, which interpolates them to the subnodes,
# or at the subnodes themselves (see _disk.NodalDisk)
SAMPLINGS = ('rnodes', 'subrings')
SAMPLING_DEFAULT = 'rnodes'


@dataclasses.dataclass(frozen=True)
class VPTDirection:
    """
    A direction of the speeds of the velocity polar traits: its uid in the
    native kernels, and the name of the speed parameter of the traits
    along it.
    """
    uid: int
    speed: str


# The directions of the speeds of the velocity polar traits, by name (see
# the module docstring of the traits)
VPT_DIRECTIONS = dict(
    tan=VPTDirection(VPT_DIRECTION_TAN, 'vt'),
    rad=VPTDirection(VPT_DIRECTION_RAD, 'vr'),
    ver=VPTDirection(VPT_DIRECTION_VER, 'vv'),
    los=VPTDirection(VPT_DIRECTION_LOS, 'vl'),
    out=VPTDirection(VPT_DIRECTION_OUT, 'vo'),
    away=VPTDirection(VPT_DIRECTION_AWAY, 'va'))

# The directions of the dispersions of the velocity dispersion polar
# traits, by name: their uids in the native kernels (see the module
# docstring of the traits)
DPT_DIRECTIONS = dict(
    iso=DPT_DIRECTION_ISO,
    rad=DPT_DIRECTION_RAD,
    tan=DPT_DIRECTION_TAN,
    ver=DPT_DIRECTION_VER)
DPT_DIRECTION_DEFAULT = 'iso'

# The azimuthal means split the circles around narrow structure at
# distances along them of at least this fraction of its size, and up to
# this multiple of the larger of its size and its distance (see
# splits_around)
_SPLIT_FINEST = 2.0 ** -8
_SPLIT_WIDEST = 8


def splits_around(
        r: np.ndarray, t: np.ndarray, size: np.ndarray, distance: np.ndarray
) -> np.ndarray:
    """
    Return the azimuths (degrees) at which the azimuthal means split the
    circles of the radii r around structures at the azimuths t (degrees),
    of the sizes size (arcsec, along the circles), whose centres are at
    the distances distance (arcsec) from the circles: at t (where a cusp
    can be), and on both sides at the distances along the circle (up to
    half a turn) that double from the larger of the distance and
    _SPLIT_FINEST times the size up to _SPLIT_WIDEST times the larger of
    the size and the distance (on a circle, a structure varies on the
    larger of its size and its distance). Between two, the structure is
    smooth. t, size and distance are arrays of shape (len(r), n); the
    result is of shape (len(r), m), NaN where a circle has fewer splits.
    """
    size = np.abs(size)
    finest = np.maximum(distance, _SPLIT_FINEST * size)
    widest = _SPLIT_WIDEST * np.maximum(size, distance)
    levels = int(np.ceil(np.log2(_SPLIT_WIDEST / _SPLIT_FINEST))) + 1
    lengths = finest[..., None] * 2.0 ** np.arange(levels)
    r = np.asarray(r, dtype=float)[:, None, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.rad2deg(lengths / r)
    w = np.where((lengths <= 2 * widest[..., None]) & (w < 180), w, np.nan)
    t = np.asarray(t, dtype=float)[..., None]
    return np.concatenate((t, t - w, t + w), axis=-1).reshape(len(r), -1)


class TraitOption(abc.ABC):
    """An option of a trait, which the trait holds (see Trait)."""

    @abc.abstractmethod
    def dump(self) -> dict[str, Any]:
        """
        Return its options in the configuration of its trait (none at its
        default).
        """

    def consts(self) -> tuple[float, ...]:
        """
        Return its constants that the kernels read, after those of the
        options before it (see Trait.consts); none here.
        """
        return ()


@dataclasses.dataclass(frozen=True)
class Zmax(TraitOption):
    """
    The option zmax of a density height trait, a vertical distortion
    trait, or a velocity height trait 'mass': a bound along z (arcsec,
    positive; see RHTrait and ZPTrait).

    Raises
    ------
    ConfigError
        If zmax is not a positive length.
    """
    value: float

    def __post_init__(self) -> None:
        zmax = self.value
        if isinstance(zmax, bool) or not (
                isinstance(zmax, numbers.Real) and 0 < zmax < np.inf):
            raise ConfigError(
                f"zmax must be a positive length; it is {zmax!r}")
        object.__setattr__(self, 'value', float(zmax))

    def dump(self) -> dict[str, Any]:
        return dict(zmax=self.value)


@dataclasses.dataclass(frozen=True)
class Direction(TraitOption):
    """
    The option direction of a velocity polar trait: the direction of its
    speed (see VPT_DIRECTIONS).

    Raises
    ------
    ConfigError
        If the direction is not one of VPT_DIRECTIONS.
    """
    name: str

    def __post_init__(self) -> None:
        if self.name not in VPT_DIRECTIONS:
            raise ConfigError(
                f"direction must be one of {list(VPT_DIRECTIONS)}; "
                f"it is {self.name!r}")

    def dump(self) -> dict[str, Any]:
        return dict(direction=self.name)


@dataclasses.dataclass(frozen=True)
class DispersionDirection(TraitOption):
    """
    The option direction of a velocity dispersion polar trait: the
    direction of its dispersion (see DPT_DIRECTIONS; by default 'iso').

    Raises
    ------
    ConfigError
        If the direction is not one of DPT_DIRECTIONS.
    """
    name: str = DPT_DIRECTION_DEFAULT

    def __post_init__(self) -> None:
        if self.name not in DPT_DIRECTIONS:
            raise ConfigError(
                f"direction must be one of {list(DPT_DIRECTIONS)}; "
                f"it is {self.name!r}")

    def uid(self) -> int:
        """Return the identifier of the direction in the native kernels."""
        return DPT_DIRECTIONS[self.name]

    def dump(self) -> dict[str, Any]:
        return {} if self.name == DPT_DIRECTION_DEFAULT \
            else dict(direction=self.name)


@dataclasses.dataclass(frozen=True)
class Sampling(TraitOption):
    """
    The option sampling of a trait with node-wise parameters: where their
    values are given (see SAMPLINGS; by default 'rnodes').

    Raises
    ------
    ConfigError
        If the sampling is not one of SAMPLINGS.
    """
    name: str = SAMPLING_DEFAULT

    def __post_init__(self) -> None:
        if self.name not in SAMPLINGS:
            raise ConfigError(
                f"sampling must be one of {list(SAMPLINGS)}; "
                f"it is '{self.name}'")

    def dump(self) -> dict[str, Any]:
        return {} if self.name == SAMPLING_DEFAULT \
            else dict(sampling=self.name)


@dataclasses.dataclass(frozen=True)
class Nodes(TraitOption):
    """
    The option rnodes of a height trait: whether its parameters are
    node-wise (by default not).
    """
    rnodes: bool = False

    def dump(self) -> dict[str, Any]:
        return dict(rnodes=True) if self.rnodes else {}


@dataclasses.dataclass(frozen=True)
class Blobs(TraitOption):
    """
    The option nblobs of a mixture trait: the number of its blobs, which
    are the rows of a table of its parameters (see ParamTable), whose
    empty row is a blob of amplitude 0: a fitter that varies the number
    of rows can drop blobs.

    Raises
    ------
    ConfigError
        If nblobs is less than 1.
    """
    nblobs: int
    table: ParamTable = dataclasses.field(
        init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.nblobs >= 1:
            raise ConfigError(
                f"nblobs must be at least 1; it is {self.nblobs}")
        object.__setattr__(self, 'table', ParamTable('blob', dict(a=0)))

    def dump(self) -> dict[str, Any]:
        return dict(nblobs=self.nblobs)

    def consts(self) -> tuple[float, ...]:
        return (self.nblobs,)

    def params(self, shape: bool) -> dict[str, ParamDesc]:
        """
        Return the parameters of the blobs: the columns of their table
        (with the shape b of the 7-parameter mixtures).
        """
        names = ('r', 't', 'a', 's', 'b', 'q', 'p') if shape \
            else ('r', 't', 'a', 's', 'q', 'p')
        return {
            name: ParamVectorDesc(self.nblobs, column=ParamColumn(
                self.table, name))
            for name in names}

    def azimuthal_edges(
            self, r: np.ndarray, params: dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Return the azimuths at which the azimuthal means split the circles
        of the radii r around the blobs (see PTrait.azimuthal_edges): by
        the smaller scale of each, at the distance of its centre from each
        circle.
        """
        size = np.abs(params['s']) * np.minimum(1, np.abs(params['q']))
        shape = (len(r), self.nblobs)
        distance = np.abs(np.asarray(r, dtype=float)[:, None] - params['r'])
        return splits_around(
            r, np.broadcast_to(params['t'], shape),
            np.broadcast_to(size, shape), distance)


@dataclasses.dataclass(frozen=True)
class Harmonic(TraitOption):
    """
    The option order of a harmonic trait: the order m of its harmonic, a
    cos(m (theta - p)), whose pattern repeats every 360 / m degrees (order
    0: axisymmetric).

    Raises
    ------
    ConfigError
        If the order is not an integer of at least 0.
    """
    order: int

    def __post_init__(self) -> None:
        order = self.order
        if isinstance(order, bool) or not isinstance(order, numbers.Integral) \
                or order < 0:
            raise ConfigError(
                f"order must be an integer of at least 0; it is {order!r}")

    def dump(self) -> dict[str, Any]:
        return dict(order=self.order)

    def consts(self) -> tuple[float, ...]:
        return (self.order,)

    def params(self, nnodes: int) -> dict[str, ParamDesc]:
        """
        Return the node-wise parameters of the harmonic: a, the amplitude,
        and p (but for order 0), the azimuth of a maximum.
        """
        params = dict(a=ParamVectorDesc(nnodes))
        if self.order > 0:
            params.update(p=ParamVectorDesc(nnodes))
        return params


@dataclasses.dataclass(frozen=True)
class Spiral(TraitOption):
    """
    A spiral density wave of m arms (the option order) in the gas of a
    disk: the options, parameters and wavenumber of the spiral traits (see
    VPTraitSpiral and RPTraitSpiral).

    The wave is the response of a gas of the sound speed c to a
    perturbation of the relative density e (epsilon) rotating at the
    pattern speed o (Omega_p), in the rotation V(R) of the disk, in the
    linear theory of tightly wound arms (Davies et al. 2009, ApJ 702,
    114, appendix). Its density is e rho_0 cos(phi), phi = m (theta - p) -
    f(R), which peaks on the arms m (theta - p) = f(R) + 2 pi n: f(R) is
    the integral from the first node of the radial wavenumber k = df/dR
    of the arms, whose azimuth changes outwards by k / m (radians per
    arcsec). Omega = V / R and kappa^2 = 2 (V / R) (dV/dR + V / R) are
    the angular and the epicyclic frequency of the rotation.

    k follows from the dispersion relation of a gas without its own
    gravity (the option wavenumber 'dispersion', the default), k^2 c^2 =
    m^2 (Omega - o)^2 - kappa^2: there is no wave where k^2 < 0 (between
    the Lindblad resonances, m^2 (Omega - o)^2 < kappa^2), where the disk
    does not rotate, and at its centre. The arms trail (the option
    winding 'trailing', the default) or lead ('leading') the rotation of
    the disk: their outer parts lag behind it or run ahead of it, k of
    the sign of -Omega or of Omega (theta increases in the direction of a
    positive rotation). Or k is given (the wavenumber 'given'), of any
    shape: a node-wise parameter at the subnodes (e.g. a formula of
    them: k = -m / (R tan(a)) winds trailing arms of the pitch angle a
    into a logarithmic spiral in a positive rotation), whose sign winds
    the arms; there is no wave at the centre.

    The rotation V(R) is that of the disk at its midplane (the azimuthal
    mean of the sum of its tangential velocity traits, parametric or
    'mass', see _profiles.rotation), which the dispersion relation and
    the velocities of the wave take: the disk must then have tangential
    velocity traits. Its galaxy computes the tables of the wave at each
    evaluation, at the subnodes of the disk, with dV/dR by the
    second-order three-point formula and f by the trapezoid rule; they
    are interpolated linearly between them, so that the subnodes must
    follow f where the arms wind tightly (rstep). The density and the
    velocities of one wave are two traits, which share its parameters
    (e.g. tied by expressions, 'disk_vpt1_p': 'disk_pt1_p').

    Its parameters: p (degrees), the azimuth of the arms where f = 0; c
    (km/s, positive), with the wavenumber of the dispersion relation; e;
    o (km/s per arcsec, positive in the direction of a positive
    rotation), with the wavenumber of the dispersion relation or for the
    velocities; and k (1/arcsec, node-wise at the subnodes), with a given
    wavenumber. The tables of the wave are not parameters (see
    Trait.wave_params).

    Raises
    ------
    ConfigError
        If the order is less than 1, the wavenumber or the winding is not
        one of theirs, or a given wavenumber has a winding.
    """
    order: int
    winding: str | None = None
    wavenumber: str = 'dispersion'

    def __post_init__(self) -> None:
        order = self.order
        if isinstance(order, bool) or not isinstance(order, numbers.Integral) \
                or order < 1:
            raise ConfigError(
                f"order is the number of arms, an integer of at least 1; it "
                f"is {order!r}")
        if self.wavenumber not in ('dispersion', 'given'):
            raise ConfigError(
                f"wavenumber must be 'dispersion' or 'given'; it is "
                f"{self.wavenumber!r}")
        if self.winding not in (None, 'trailing', 'leading'):
            raise ConfigError(
                f"winding must be 'trailing' or 'leading'; it is "
                f"{self.winding!r}")
        if self.wavenumber == 'given' and self.winding is not None:
            raise ConfigError(
                "a given wavenumber winds the arms by its sign: winding is "
                "for the wavenumber of the dispersion relation")

    def dump(self) -> dict[str, Any]:
        info: dict[str, Any] = dict(order=self.order)
        if self.wavenumber != 'dispersion':
            info['wavenumber'] = self.wavenumber
        if self.winding not in (None, 'trailing'):
            info['winding'] = self.winding
        return info

    def consts(self, nparams_sm: int) -> tuple[float, ...]:
        """
        Return the constants of its trait of nparams_sm parameters that
        are not node-wise (the kernels read the tables after them).
        """
        return self.order, nparams_sm

    def params_sm(self, pattern_speed: bool) -> dict[str, ParamDesc]:
        """
        Return its parameters that are not node-wise, p first (the kernels
        read it there): c with the wavenumber of the dispersion relation,
        and o with it or if pattern_speed.
        """
        dispersion = self.wavenumber == 'dispersion'
        params = dict(p=ParamScalarDesc())
        if dispersion:
            params['c'] = ParamScalarDesc()
        params['e'] = ParamScalarDesc()
        if dispersion or pattern_speed:
            params['o'] = ParamScalarDesc()
        return params

    def params_rnw(
            self, tables: Sequence[str], nnodes: int
    ) -> dict[str, ParamDesc]:
        """
        Return its node-wise parameters: its tables (see
        Trait.wave_params), first (the kernels read them there), and k
        with a given wavenumber.
        """
        names = list(tables) + (['k'] if self.wavenumber == 'given' else [])
        return {name: ParamVectorDesc(nnodes) for name in names}

    def wavenumber_at(
            self,
            r: np.ndarray,
            v: np.ndarray,
            dv: np.ndarray,
            params: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return Omega, kappa^2 and k of the wave at the radii r (arcsec,
        increasing: the subnodes of its disk), in the rotation v of its
        disk there (km/s) of the derivative dv (km/s per arcsec), for the
        values of its parameters (c and o, or k), and where there is a
        wave (k is 0 elsewhere).
        """
        r = np.asarray(r, dtype=float)
        rotating = (r > 0) & (v != 0)
        omega = np.divide(v, r, out=np.zeros_like(r), where=rotating)
        kappa2 = 2 * omega * (dv + omega)
        if self.wavenumber == 'given':
            waves = r > 0
            k = np.where(waves, np.asarray(params['k'], dtype=float), 0)
            return omega, kappa2, k, waves
        m = self.order
        c, o = float(params['c']), float(params['o'])
        k2 = (m ** 2 * (omega - o) ** 2 - kappa2) / c ** 2
        waves = rotating & (k2 > 0)
        sense = 1 if self.winding == 'leading' else -1
        k = sense * np.sign(omega) * np.sqrt(np.where(waves, k2, 0))
        return omega, kappa2, k, waves

    @staticmethod
    def phase(r: np.ndarray, k: np.ndarray) -> np.ndarray:
        """
        Return f, the integral of k from the first of the radii r (arcsec,
        increasing) to each, by the trapezoid rule.
        """
        steps = np.diff(r) * (k[1:] + k[:-1]) / 2
        return np.concatenate(([0], np.cumsum(steps)))
