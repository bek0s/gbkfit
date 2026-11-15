"""
The radial profiles of the nodal traits of a component (a disk, a cone,
or the traits of a spheroid; see NodalTraited): the azimuthal means of
its polar traits, each times its height trait at its midplane (or the
mid-surface of the walls of a cone), at radii: the sum of its density
traits where its selection traits keep it, of its velocity traits along
a direction, and of its dispersion traits (see disks._profiles, for what
a disk adds: its gravity and its vertical equilibrium).
"""

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

from gbkfit.params import ParamValues
from .traits import DPT_DIRECTION_DEFAULT, Trait


__all__ = [
    'NodalTraited',
    'TraitsPlan',
    'at_heights',
    'at_midplane',
    'axisymmetric',
    'azimuthal_edges',
    'azimuthal_mean',
    'directional_sums',
    'isotropic',
    'polar_densities',
    'selected',
    'trait_density',
    'trait_dispersion',
    'trait_speed'
]


# The azimuthal means of traits that are not axisymmetric: the trapezoid
# rule on at least and at most these numbers of azimuths, doubled until
# two means agree to the tolerance, relative to the largest value
_NAZIMUTHS_MIN = 64
_NAZIMUTHS_MAX = 4096
_AZIMUTHAL_TOLERANCE = 1e-6

# The azimuthal means of traits that split the circles (where they jump,
# and around their narrow structure): Gauss-Legendre quadrature on each
# piece between two splits, of at least and at most these numbers of
# points, doubled until two means agree to the tolerance
_NPOINTS_MIN = 4
_NPOINTS_MAX = 256


class NodalTraited(Protocol):
    """
    A component of traits on radial nodes (see _nodal.NodalTraits): a
    disk, a cone, or the traits of a spheroid.
    """

    def traits(self, kind: str) -> tuple[Trait, ...]:
        """Return its traits of one kind (see the kinds of its traits)."""
        ...

    def trait_values(
            self, kind: str, params: ParamValues
    ) -> tuple[dict[str, np.ndarray], ...]:
        """
        Return the values of the parameters of each trait of one kind, by
        their names in the trait, from the values of its parameters: the
        node-wise ones at its subnodes.
        """
        ...

    def subrnodes(self) -> Sequence[float]:
        """Return the radii of its subnodes (arcsec)."""
        ...


class TraitsPlan(Protocol):
    """
    A plan of a NodalTraited, which evaluates its traits at points, as
    its kernels do.
    """

    def evaluate_polar_traits(
            self, params: ParamValues, kind: str, r: npt.ArrayLike,
            theta: npt.ArrayLike
    ) -> np.ndarray:
        """
        Return each polar trait of one kind at the points (r, theta)
        (arcsec, degrees): a row for each trait, of the shape of the
        points; the velocity traits as their speeds, each along its
        direction.
        """
        ...

    def evaluate_height_traits(
            self, params: ParamValues, kind: str, r: npt.ArrayLike,
            z: npt.ArrayLike
    ) -> np.ndarray:
        """
        Return each height trait of one kind at the points (r, z), z the
        height above the midplane (arcsec): a row for each trait, of the
        shape of the points.
        """
        ...


def azimuthal_mean(
        values: Callable[[np.ndarray], np.ndarray],
        axisymmetric: bool,
        edges: np.ndarray | None = None
) -> np.ndarray:
    """
    Return the mean over the azimuth of values(theta), an array of shape
    (rows, radii, len(theta)) at the azimuths theta (degrees; an array of
    shape (radii, n) with edges), as an array of shape (rows, radii). If
    axisymmetric (the values are the same at every azimuth), their values
    at one azimuth, exactly. With edges, the azimuths (degrees) at which
    to split the circles (see _azimuthal_mean_pieces). Otherwise by the
    trapezoid rule, exact for harmonics of orders below the number of
    azimuths, to _AZIMUTHAL_TOLERANCE of the largest absolute value of
    each row: structure narrower than the spacing of the azimuths (2 pi R
    / 128 at least) can go unseen by two successive numbers of azimuths,
    and be missed.

    Raises
    ------
    ValueError
        If the mean does not converge on _NAZIMUTHS_MAX azimuths.
    """
    if axisymmetric:
        return values(np.zeros(1))[..., 0]
    if edges is not None:
        return _azimuthal_mean_pieces(values, edges)
    count = _NAZIMUTHS_MIN
    sampled = values(np.arange(count) * 360 / count)
    mean = sampled.mean(axis=-1)
    largest = np.max(np.abs(sampled), axis=(-2, -1), initial=0)
    while count < _NAZIMUTHS_MAX:
        sampled = values((np.arange(count) + 0.5) * 360 / count)
        largest = np.maximum(
            largest, np.max(np.abs(sampled), axis=(-2, -1), initial=0))
        refined = (mean + sampled.mean(axis=-1)) / 2
        change = np.max(np.abs(refined - mean), axis=-1, initial=0)
        mean = refined
        count *= 2
        if np.all(change <= _AZIMUTHAL_TOLERANCE * largest):
            return mean
    raise ValueError(
        f"the azimuthal mean of the traits does not converge "
        f"on {_NAZIMUTHS_MAX} azimuths: its azimuthal structure is too fine "
        f"or too sharp (e.g. narrow blobs, or the edges of selection "
        f"traits)")


def _azimuthal_mean_pieces(
        values: Callable[[np.ndarray], np.ndarray], edges: np.ndarray
) -> np.ndarray:
    """
    Return the mean over the azimuth of values(theta) (see
    azimuthal_mean), which are smooth between the azimuths edges
    (degrees, an array of shape (radii, n), NaN where a radius has fewer;
    see traits.PTrait.azimuthal_edges): by Gauss-Legendre quadrature on
    each piece between two of them (exact for polynomials of degree below
    twice its number of points), of _NPOINTS_MIN points, doubled until two
    means agree to _AZIMUTHAL_TOLERANCE of the largest mean absolute value
    of each row (its light, also for narrow structure, whose peak is far
    above its mean).

    Raises
    ------
    ValueError
        If the mean does not converge on _NPOINTS_MAX points per piece.
    """
    # The pieces of each circle, from its first edge to the same one a turn
    # later (from 0 to 360 without edges), and pieces of no length for the
    # missing edges of a radius
    edges = np.sort(np.mod(edges, 360), axis=-1)
    first = np.where(np.isnan(edges[:, :1]), 0, edges[:, :1])
    bounds = np.sort(np.concatenate(
        (first, np.where(np.isnan(edges), first + 360, edges), first + 360),
        axis=-1), axis=-1)
    lo, hi = bounds[:, :-1, None], bounds[:, 1:, None]

    # TODO: speed. The values at every point are copied from the device
    # and summed here (most of the time: e.g. 14 ms on CUDA for ten blobs
    # on 200 circles, against 7 ms with the trapezoid rule); a kernel that
    # evaluates the traits and sums them with the weights on the device
    # would return only the means
    def means(npoints: int) -> tuple[np.ndarray, np.ndarray]:
        # The mean, and the largest mean absolute value of each row
        x, w = np.polynomial.legendre.leggauss(npoints)
        theta = ((hi + lo) + (hi - lo) * x) / 2
        weights = ((hi - lo) * w / 2).reshape(len(lo), -1) / 360
        sampled = values(theta.reshape(len(theta), -1))
        return (np.sum(sampled * weights, axis=-1),
                np.max(np.sum(np.abs(sampled) * weights, axis=-1), axis=-1))
    npoints = _NPOINTS_MIN
    previous, _ = means(npoints)
    while npoints < _NPOINTS_MAX:
        npoints *= 2
        current, scale = means(npoints)
        change = np.max(np.abs(current - previous), axis=-1, initial=0)
        if np.all(change <= _AZIMUTHAL_TOLERANCE * scale):
            return current
        previous = current
    raise ValueError(
        f"the azimuthal mean of the traits does not converge "
        f"on {_NPOINTS_MAX} points between each two of its splits: its "
        f"azimuthal structure is too fine or too sharp")


def azimuthal_edges(
        traited: NodalTraited,
        params: ParamValues,
        kinds: Sequence[str],
        r: np.ndarray
) -> np.ndarray | None:
    """
    Return the azimuths (degrees) at which the polar traits of the given
    kinds of a NodalTraited split the circles of the radii r (arcsec), all
    together: an array of shape (len(r), n), NaN where a circle has fewer;
    None if all of them are smooth (see traits.PTrait.azimuthal_edges).
    """
    edges = [
        trait_edges
        for kind in kinds if traited.traits(kind)
        for trait, values in zip(
            traited.traits(kind), traited.trait_values(kind, params))
        if (trait_edges := trait.azimuthal_edges(
            r, values, traited.subrnodes())) is not None]
    if not edges:
        return None
    # (without the columns of no edge at any radius, which cost points)
    edges = np.sort(np.concatenate(edges, axis=1), axis=1)
    return edges[:, :np.max(np.sum(~np.isnan(edges), axis=1), initial=0)]


def axisymmetric(traited: NodalTraited, kinds: Sequence[str]) -> bool:
    """
    Check whether the polar traits of the given kinds of a NodalTraited
    all have the same value at every azimuth (see RPTrait.axisymmetric).
    """
    return all(
        trait.axisymmetric()
        for kind in kinds for trait in traited.traits(kind))


def selected(
        traited: NodalTraited, plan: TraitsPlan, params: ParamValues,
        r: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """
    Return whether the selection traits of a NodalTraited keep each point
    (r, theta) of it (all if it has none), as an array of their shape.
    """
    if not traited.traits('spt'):
        return np.ones(np.broadcast_shapes(r.shape, theta.shape), dtype=bool)
    return plan.evaluate_polar_traits(
        params, 'spt', r, theta).sum(axis=0) != 0


def at_midplane(
        traited: NodalTraited, plan: TraitsPlan, params: ParamValues,
        kind: str, r: np.ndarray
) -> np.ndarray:
    """
    Return the height traits of one kind of a NodalTraited at z = 0 and
    the radii r (arcsec, an array of shape (radii, 1)), which multiply the
    polar traits they pair with: an array of shape (traits, radii, 1),
    ones without them.
    """
    if not traited.traits(kind):
        return np.ones((1, len(r), 1))
    return plan.evaluate_height_traits(params, kind, r, np.zeros(1))


def at_heights(
        traited: NodalTraited, plan: TraitsPlan, params: ParamValues,
        kind: str, r: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """
    Return the height traits of one kind of a NodalTraited at the radii r
    (arcsec, an array of shape (radii, 1)) and the heights z (arcsec),
    which multiply the polar traits they pair with: an array of shape
    (traits, len(z), radii, 1), ones without them.
    """
    if not traited.traits(kind):
        return np.ones((1, 1, len(r), 1))
    values = plan.evaluate_height_traits(params, kind, r, z)
    return np.moveaxis(np.asarray(values, dtype=float), -1, 1)[..., None]


def polar_densities(
        traited: NodalTraited, plan: TraitsPlan, params: ParamValues,
        r: np.ndarray
) -> np.ndarray:
    """
    Return the azimuthal mean of each density polar trait of a
    NodalTraited where its selection traits keep it, at the radii r
    (arcsec): an array of shape (traits, len(r)).
    """
    r = r[:, None]

    def values(theta: np.ndarray) -> np.ndarray:
        out = np.asarray(
            plan.evaluate_polar_traits(params, 'rpt', r, theta), dtype=float)
        return out * selected(traited, plan, params, r, theta)
    kinds = ('rpt', 'spt')
    return azimuthal_mean(
        values, axisymmetric(traited, kinds),
        azimuthal_edges(traited, params, kinds, r[:, 0]))


def trait_speed(
        traited: NodalTraited,
        plan: TraitsPlan,
        params: ParamValues,
        radii: npt.ArrayLike,
        direction: str
) -> np.ndarray:
    """
    Return the azimuthal mean of the sum of the velocity traits of a
    NodalTraited along a direction, each times its velocity height trait
    at z = 0, at the radii (arcsec), of the values of its parameters (with
    those its galaxy gives its traits, if any); 0 without them.
    """
    r = np.asarray(radii, dtype=float)[:, None]
    along = [
        trait.direction() == direction for trait in traited.traits('vpt')]
    if not any(along):
        return np.zeros(len(r))
    heights = at_midplane(traited, plan, params, 'vht', r)

    def values(theta: np.ndarray) -> np.ndarray:
        speeds = plan.evaluate_polar_traits(params, 'vpt', r, theta) \
            * heights
        return np.sum(speeds[along], axis=0, keepdims=True, dtype=float)
    # (the other traits, e.g. a spiral wave, do not make its sum vary)
    smooth = all(
        trait.axisymmetric()
        for trait, kept in zip(traited.traits('vpt'), along) if kept)
    return azimuthal_mean(
        values, smooth,
        azimuthal_edges(traited, params, ('vpt',), r[:, 0]))[0]


def trait_density(
        traited: NodalTraited, plan: TraitsPlan, params: ParamValues,
        r: npt.ArrayLike
) -> np.ndarray:
    """
    Return the azimuthal mean of the sum of the density polar traits of a
    NodalTraited where its selection traits keep it (its density per unit
    area: of the plane of a disk, of the mid-surface of the walls of a
    cone), at the radii r (arcsec).
    """
    return polar_densities(
        traited, plan, params, np.asarray(r, dtype=float)).sum(axis=0)


def isotropic(traited: NodalTraited) -> bool:
    """
    Check whether the dispersion traits of a NodalTraited are all
    isotropic (see traits.DPTrait.direction).
    """
    return all(
        trait.direction() == DPT_DIRECTION_DEFAULT
        for trait in traited.traits('dpt'))


def directional_sums(
        traited: NodalTraited, values: np.ndarray,
        directions: Sequence[str]
) -> list[np.ndarray]:
    """
    Return the sums of the dispersion traits of a NodalTraited along each
    of the given directions ('rad', 'tan' or 'ver'; or the default,
    isotropic), from their values (a row for each trait, in their order):
    those of its isotropic traits and those along the direction.
    """
    result = []
    for direction in directions:
        rows = [
            i for i, trait in enumerate(traited.traits('dpt'))
            if trait.direction() in (DPT_DIRECTION_DEFAULT, direction)]
        result.append(np.sum(values[rows], axis=0, dtype=float))
    return result


def trait_dispersion(
        traited: NodalTraited, plan: TraitsPlan, params: ParamValues,
        radii: npt.ArrayLike, direction: str | None = None
) -> np.ndarray:
    """
    Return the velocity dispersion of the traits of a NodalTraited at its
    midplane (km/s), at the radii (arcsec): the azimuthal mean of the
    absolute sum of its dispersion polar traits, each times its height
    trait there; along a direction, if given ('rad', 'tan' or 'ver': its
    isotropic traits and those along it), as it must be if they are not
    all isotropic; 0 without them.

    Raises
    ------
    ValueError
        If the direction is not valid, or not given for traits that are
        not all isotropic.
    """
    if direction is None and not isotropic(traited):
        raise ValueError(
            "the dispersion is not isotropic: give its direction, 'rad', "
            "'tan' or 'ver'")
    if direction not in (None, 'rad', 'tan', 'ver'):
        raise ValueError(
            f"direction must be 'rad', 'tan' or 'ver'; it is {direction!r}")
    r = np.asarray(radii, dtype=float)[:, None]
    if not traited.traits('dpt'):
        return np.zeros(len(r))
    heights = at_midplane(traited, plan, params, 'dht', r)

    def values(theta: np.ndarray) -> np.ndarray:
        sums = directional_sums(
            traited,
            plan.evaluate_polar_traits(params, 'dpt', r, theta) * heights,
            (direction or DPT_DIRECTION_DEFAULT,))
        return np.abs(sums[0])[None]
    kinds = ('dpt',)
    return azimuthal_mean(
        values, axisymmetric(traited, kinds),
        azimuthal_edges(traited, params, kinds, r[:, 0]))[0]
