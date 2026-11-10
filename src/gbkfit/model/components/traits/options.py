"""
The options of the traits: small values that check themselves and give
their configuration (see TraitOption), which the traits hold (see
Trait). A configuration gives them as flat options of its trait (e.g.
direction: tan, order: 1, sampling: subrings).
"""

import abc
import dataclasses
from collections.abc import Sequence
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np

from gbkfit.config import Choices
from gbkfit.params.pdescs import (
    ParamColumn, ParamDesc, ParamScalarDesc, ParamTable, ParamVectorDesc)


__all__ = [
    'DPT_DIRECTIONS',
    'DPT_DIRECTION_DEFAULT',
    'SAMPLINGS',
    'SAMPLING_DEFAULT',
    'SAMPLING_TAIL',
    'VPT_DIRECTIONS',
    'Blobs',
    'DPTDirectionChoice',
    'Direction',
    'DispersionDirection',
    'Harmonic',
    'Nodewise',
    'Sampling',
    'SamplingChoice',
    'Spiral',
    'SpiralWavenumber',
    'SpiralWinding',
    'TraitOption',
    'VPTDirection',
    'VPTDirectionChoice',
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
SAMPLINGS = ('rings', 'subrings')
SAMPLING_DEFAULT = 'rings'

# The option of a sampling (see SAMPLINGS)
SamplingChoice: TypeAlias = Annotated[Literal[SAMPLINGS], Choices(
    rings='At the rings', subrings='At the subrings')]
# Where the values that the galaxy of a disk gives beyond its last ring
# too are given (e.g. the rotation of its mass, see VPTraitMass; not an
# option of the traits): at the subnodes, and beyond the last, at the
# radii of its tail (see _nodal.tail_radii)
SAMPLING_TAIL = 'subrings_tail'


@dataclasses.dataclass(frozen=True)
class VPTDirection:
    """
    A direction of the speeds of the velocity polar traits: its uid in the
    native kernels, its title, and the name and the title of the speed
    parameter of the traits along it.
    """
    uid: int
    title: str
    speed: str
    speed_title: str


# The directions of the speeds of the velocity polar traits, by name (see
# the module docstring of the traits)
VPT_DIRECTIONS = dict(
    tan=VPTDirection(VPT_DIRECTION_TAN, 'Tangential', 'vt', 'Rotation speed'),
    rad=VPTDirection(VPT_DIRECTION_RAD, 'Radial', 'vr', 'Radial speed'),
    ver=VPTDirection(VPT_DIRECTION_VER, 'Vertical', 'vv', 'Vertical speed'),
    los=VPTDirection(
        VPT_DIRECTION_LOS, 'Along the line of sight', 'vl',
        'Line-of-sight speed'),
    out=VPTDirection(
        VPT_DIRECTION_OUT, 'Outward, from the centre', 'vo',
        'Outward speed'),
    away=VPTDirection(
        VPT_DIRECTION_AWAY, 'Away from the plane', 'va',
        'Speed away from the plane'))

# The option of the direction of a velocity polar trait
VPTDirectionChoice: TypeAlias = Annotated[
    Literal[tuple(VPT_DIRECTIONS)],
    Choices({name: d.title for name, d in VPT_DIRECTIONS.items()})]

# The directions of the dispersions of the velocity dispersion polar
# traits, by name: their uids in the native kernels (see the module
# docstring of the traits)
DPT_DIRECTIONS = dict(
    iso=DPT_DIRECTION_ISO,
    rad=DPT_DIRECTION_RAD,
    tan=DPT_DIRECTION_TAN,
    ver=DPT_DIRECTION_VER)
DPT_DIRECTION_DEFAULT = 'iso'

# The option of the direction of a velocity dispersion polar trait
DPTDirectionChoice: TypeAlias = Annotated[
    Literal[tuple(DPT_DIRECTIONS)], Choices(
        iso='Isotropic', rad='Radial', tan='Tangential', ver='Vertical')]

# The options of the winding and of the wavenumber of a spiral density
# wave (see Spiral)
SpiralWinding: TypeAlias = Annotated[
    Literal['trailing', 'leading'],
    Choices(trailing='Trailing arms', leading='Leading arms')]
SpiralWavenumber: TypeAlias = Annotated[
    Literal['dispersion', 'given'],
    Choices(dispersion='From the dispersion relation', given='Given (k)')]

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
    ValueError
        If zmax is not a positive length.
    """
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, 'value', float(self.value))

    def dump(self) -> dict[str, Any]:
        return dict(zmax=self.value)


@dataclasses.dataclass(frozen=True)
class Direction(TraitOption):
    """
    The option direction of a velocity polar trait: the direction of its
    speed (see VPT_DIRECTIONS).

    Raises
    ------
    ValueError
        If the direction is not one of VPT_DIRECTIONS.
    """
    name: str

    def dump(self) -> dict[str, Any]:
        return dict(direction=self.name)


@dataclasses.dataclass(frozen=True)
class DispersionDirection(TraitOption):
    """
    The option direction of a velocity dispersion polar trait: the
    direction of its dispersion (see DPT_DIRECTIONS; by default 'iso').

    Raises
    ------
    ValueError
        If the direction is not one of DPT_DIRECTIONS.
    """
    name: str = DPT_DIRECTION_DEFAULT

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
    values are given (see SAMPLINGS; by default 'rings').

    Raises
    ------
    ValueError
        If the sampling is not one of SAMPLINGS.
    """
    name: str = SAMPLING_DEFAULT

    def dump(self) -> dict[str, Any]:
        return {} if self.name == SAMPLING_DEFAULT \
            else dict(sampling=self.name)


@dataclasses.dataclass(frozen=True)
class Nodewise(TraitOption):
    """
    The option nodewise of a height trait: whether its parameters are
    node-wise (by default not).
    """
    nodewise: bool = False

    def dump(self) -> dict[str, Any]:
        return dict(nodewise=True) if self.nodewise else {}


@dataclasses.dataclass(frozen=True)
class Blobs(TraitOption):
    """
    The option nblobs of a mixture trait: the number of its blobs, which
    are the rows of a table of its parameters (see ParamTable), whose
    empty row is a blob of amplitude 0: a fitter that varies the number
    of rows can drop blobs.

    Raises
    ------
    ValueError
        If nblobs is less than 1.
    """
    nblobs: int
    table: ParamTable = dataclasses.field(
        init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'table', ParamTable('blob', dict(a=0)))

    def dump(self) -> dict[str, Any]:
        return dict(nblobs=self.nblobs)

    def consts(self) -> tuple[float, ...]:
        return (self.nblobs,)

    def params(
            self, shape: dict[str, Any] | None, unit: str | None
    ) -> dict[str, ParamDesc]:
        """
        Return the parameters of the blobs: the columns of their table
        (with the shape b of the 7-parameter mixtures, of the given valid
        range, e.g. dict(min=0); None for none), of the amplitudes a of the
        given unit (None for a density, whose component knows it).
        """
        titles = dict(
            r='Centre radius', t='Centre azimuth', a='Amplitude',
            s='Scale length', b='Shape', q='Axis ratio', p='Orientation')
        descs = dict(
            r='the radius of the centre of each blob',
            t='the azimuth of the centre of each blob',
            a='the amplitude of each blob',
            s='the scale of each blob along its major axis',
            b='the shape of the profile of each blob',
            q='the axis ratio of each blob (minor over major)',
            p='the angle of the major axis of each blob from the '
              'tangential direction')
        units = dict(
            r='arcsec', t='deg', a=unit, s='arcsec', b='', q='', p='deg')
        # (amplitudes 0: those of the empty row, see Blobs)
        defaults = dict(r=1, t=0, a=0, s=0.2, b=2, q=1, p=0)
        # (their valid ranges and periods: an elliptical blob is the same
        # turned by 180 degrees)
        valid = dict(
            r=dict(min=0), t=dict(period=360), s=dict(min=0), b=shape,
            q=dict(min=0), p=dict(period=180))
        names = ('r', 't', 'a', 's', 'b', 'q', 'p') if shape is not None \
            else ('r', 't', 'a', 's', 'q', 'p')
        return {
            name: ParamVectorDesc(
                self.nblobs, title=titles[name], desc=descs[name],
                unit=units[name], default=defaults[name],
                **valid.get(name, {})
            ).with_column(ParamColumn(self.table, name))
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
    ValueError
        If the order is not an integer of at least 0.
    """
    order: int

    def dump(self) -> dict[str, Any]:
        return dict(order=self.order)

    def consts(self) -> tuple[float, ...]:
        return (self.order,)

    def params(self, nnodes: int, unit: str | None) -> dict[str, ParamDesc]:
        """
        Return the node-wise parameters of the harmonic: a, the amplitude,
        of the given unit (None for a density, whose component knows it),
        and p (but for order 0), the azimuth of a maximum.
        """
        params = dict(a=ParamVectorDesc(
            nnodes, title='Amplitude', desc='the amplitude of the harmonic',
            unit=unit, default=0))
        if self.order > 0:
            # (the pattern repeats every 360 / m degrees: of a bisymmetric
            # flow (m = 2), p and p + 180 are the same)
            params.update(p=ParamVectorDesc(
                nnodes, title='Phase', desc='the azimuth of a maximum',
                unit='deg', default=0, period=360 / self.order))
        return params


# The titles and the descriptions of the tables of the spiral density
# waves, by name (see WaveInput, and the method wave of the spiral traits)
_SPIRAL_TABLES = dict(
    a=('Density amplitude',
       'the amplitude of the density of the wave, e Sigma_0'),
    ar=('Radial amplitude',
        'the amplitude of the radial velocity of the wave'),
    at=('Tangential amplitude',
        'the amplitude of the tangential velocity of the wave'),
    f=('Radial phase', 'the phase f of the arms, the integral of k'))


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
    (e.g. tied by expressions, 'disk_vpt2_p': 'disk_rpt2_p').

    Its parameters: p (degrees), the azimuth of the arms where f = 0; c
    (km/s, positive), with the wavenumber of the dispersion relation; e;
    o (km/s per arcsec, positive in the direction of a positive
    rotation), with the wavenumber of the dispersion relation or for the
    velocities; and k (1/arcsec, node-wise at the subnodes), with a given
    wavenumber. The tables of the wave are not parameters (see
    WaveInput).

    Raises
    ------
    ValueError
        If the order is less than 1, the wavenumber or the winding is not
        one of theirs, or a given wavenumber has a winding.
    """
    order: int
    winding: str | None = None
    wavenumber: str = 'dispersion'

    def __post_init__(self) -> None:
        if self.wavenumber == 'given' and self.winding is not None:
            raise ValueError(
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
        # (m arms: the pattern repeats every 360 / m degrees)
        params = dict(p=ParamScalarDesc(
            title='Arm azimuth', desc='the azimuth of the arms where f = 0',
            unit='deg', default=0, period=360 / self.order))
        if dispersion:
            params['c'] = ParamScalarDesc(
                title='Sound speed', desc='the sound speed of the gas',
                min=0, unit='km/s', default=10)
        params['e'] = ParamScalarDesc(
            title='Relative amplitude',
            desc='the amplitude of the density of the wave, relative to '
                 'that of the disk', unit='', default=0.1)
        if dispersion or pattern_speed:
            params['o'] = ParamScalarDesc(
                title='Pattern speed',
                desc='the angular speed of the arms, positive in the '
                     'direction of a positive rotation',
                unit='km / (s arcsec)', default=0)
        return params

    def params_rnw(
            self, tables: Sequence[str], nnodes: int
    ) -> dict[str, ParamDesc]:
        """
        Return its node-wise parameters: its tables (see WaveInput),
        first (the kernels read them there), and k
        with a given wavenumber.
        """
        params = {
            name: ParamVectorDesc(
                nnodes, title=_SPIRAL_TABLES[name][0],
                desc=_SPIRAL_TABLES[name][1], default=0)
            for name in tables}
        if self.wavenumber == 'given':
            params['k'] = ParamVectorDesc(
                nnodes, title='Radial wavenumber',
                desc='the radial wavenumber of the arms, whose sign winds '
                     'them', unit='1 / arcsec', default=1)
        return params

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
