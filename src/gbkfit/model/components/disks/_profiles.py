"""
The radial profiles of a disk, as its galaxy takes them for gravity: the
azimuthal means of its traits, for the mass components that follow it
(see mass.DiskDensity), for its pressure support (see
traits.VPTraitMass), and its rms height, rotation and dispersions, for
its vertical equilibrium (see traits.DPTraitEquilibrium and
traits.RHTraitEquilibrium); its rotation, for its density waves (see
traits.VPTraitSpiral); and its rotation and dispersion, for the
functions of its galaxy (see GalaxyFunctions).
"""

import dataclasses
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from gbkfit.params import ParamValues
from .._nodal import subring_radii, tail_end, tail_radii
from ..traits import DPT_DIRECTION_DEFAULT, RPTraitSpiral, VPTraitSpiral
from ._nodal import NodalDisk, DiskPlan

if TYPE_CHECKING:
    from ._stage import DiskState


__all__ = [
    'NodalDiskDensity',
    'dispersion',
    'jeans_dispersions',
    'pressure_support',
    'rms_heights',
    'rotation',
    'surface_density'
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

# The heights of the vertical profile of a disk (see _VerticalProfile):
# Gauss-Legendre points on panels of geometric widths, from 1e-9 of the
# zmax of each density height trait to its zmax (in units of it)
_HEIGHT_EDGES = np.concatenate(([0], np.geomspace(1e-9, 1, 22)))
_HEIGHT_NPOINTS = 16

# How close to 1 the integral of a height pdf over its heights must be
_PDF_TOLERANCE = 1e-6


def _azimuthal_mean(
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
        f"the azimuthal mean of the density of a disk does not converge "
        f"on {_NAZIMUTHS_MAX} azimuths: its azimuthal structure is too fine "
        f"or too sharp (e.g. narrow blobs, or the edges of selection "
        f"traits)")


def _azimuthal_mean_pieces(
        values: Callable[[np.ndarray], np.ndarray], edges: np.ndarray
) -> np.ndarray:
    """
    Return the mean over the azimuth of values(theta) (see
    _azimuthal_mean), which are smooth between the azimuths edges
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
        f"the azimuthal mean of the density of a disk does not converge "
        f"on {_NPOINTS_MAX} points between each two of its splits: its "
        f"azimuthal structure is too fine or too sharp")


def _edges(
        disk: NodalDisk,
        params: ParamValues,
        kinds: Sequence[str],
        r: np.ndarray
) -> np.ndarray | None:
    """
    Return the azimuths (degrees) at which the polar traits of the given
    kinds of a disk split the circles of the radii r (arcsec), all
    together: an array of shape (len(r), n), NaN where a circle has fewer;
    None if all of them are smooth (see traits.PTrait.azimuthal_edges).
    """
    edges = [
        trait_edges
        for kind in kinds if disk.traits(kind)
        for trait, values in zip(
            disk.traits(kind), disk.trait_values(kind, params))
        if (trait_edges := trait.azimuthal_edges(
            r, values, disk.subrnodes())) is not None]
    if not edges:
        return None
    # (without the columns of no edge at any radius, which cost points)
    edges = np.sort(np.concatenate(edges, axis=1), axis=1)
    return edges[:, :np.max(np.sum(~np.isnan(edges), axis=1), initial=0)]


def _axisymmetric(disk: NodalDisk, kinds: Sequence[str]) -> bool:
    """
    Check whether the polar traits of the given kinds of a disk all have
    the same value at every azimuth (see RPTrait.axisymmetric).
    """
    return all(
        trait.axisymmetric() for kind in kinds for trait in disk.traits(kind))


def _isotropic(disk: NodalDisk) -> bool:
    """
    Check whether the dispersion of a disk is the same along every
    direction: its dispersion polar traits and the dispersion of its
    vertical equilibrium, if any, isotropic (see traits and
    traits.DPTraitEquilibrium).
    """
    equilibrium = disk.equilibrium()
    return all(
        trait.direction() == DPT_DIRECTION_DEFAULT
        for trait in disk.traits('dpt')) and (
            equilibrium is None
            or equilibrium.direction() == DPT_DIRECTION_DEFAULT)


def _dispersions(
        disk: NodalDisk,
        values: np.ndarray,
        balanced: np.ndarray | float,
        directions: Sequence[str]
) -> list[np.ndarray]:
    """
    Return the dispersion of a disk along each of the given directions
    ('rad', 'tan' or 'ver'; 'iso' for an isotropic disk, see _isotropic),
    from the values of its dispersion polar traits times their height
    traits (values, a row for each trait, in their order): the sum of its
    isotropic traits and those along the direction, with the dispersion of
    its vertical equilibrium (balanced) in quadrature along the directions
    it adds to (every one, or 'ver' only).
    """
    equilibrium = disk.equilibrium()
    vertical = equilibrium is not None and equilibrium.direction() == 'ver'
    result = []
    for direction in directions:
        rows = [
            i for i, trait in enumerate(disk.traits('dpt'))
            if trait.direction() in (DPT_DIRECTION_DEFAULT, direction)]
        total = np.sum(values[rows], axis=0, dtype=float)
        balances = not vertical or direction == 'ver'
        result.append(np.hypot(total, balanced) if balances else np.abs(total))
    return result


def _with_inputs(
        disk: NodalDisk,
        params: ParamValues,
        state: 'DiskState | None' = None
) -> ParamValues:
    """
    Return the values of the parameters of a disk with those that its
    galaxy gives its traits, of the state, if given (see
    NodalDisk.trait_inputs).
    """
    return disk.trait_inputs(state) | dict(params)


def _selected(
        disk: NodalDisk, plan: DiskPlan, params: ParamValues, r: np.ndarray,
        theta: np.ndarray
) -> np.ndarray:
    """
    Return whether the selection traits of a disk keep each point (r,
    theta) of it (all if it has none), as an array of their shape.
    """
    if not disk.traits('spt'):
        return np.ones(np.broadcast_shapes(r.shape, theta.shape), dtype=bool)
    return plan.evaluate_polar_traits(
        params, 'spt', r, theta).sum(axis=0) != 0


def _at_midplane(
        disk: NodalDisk, plan: DiskPlan, params: ParamValues, kind: str,
        r: np.ndarray
) -> np.ndarray:
    """
    Return the height traits of one kind of a disk at z = 0 and the radii
    r (arcsec, an array of shape (radii, 1)), which multiply the polar
    traits they pair with: an array of shape (traits, radii, 1), ones
    without them.
    """
    if not disk.traits(kind):
        return np.ones((1, len(r), 1))
    return plan.evaluate_height_traits(params, kind, r, np.zeros(1))


def _at_heights(
        disk: NodalDisk, plan: DiskPlan, params: ParamValues, kind: str,
        r: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """
    Return the height traits of one kind of a disk at the radii r (arcsec,
    an array of shape (radii, 1)) and the heights z (arcsec), which
    multiply the polar traits they pair with: an array of shape (traits,
    len(z), radii, 1), ones without them.
    """
    if not disk.traits(kind):
        return np.ones((1, 1, len(r), 1))
    values = plan.evaluate_height_traits(params, kind, r, z)
    return np.moveaxis(np.asarray(values, dtype=float), -1, 1)[..., None]


def _surface_densities(
        disk: NodalDisk, plan: DiskPlan, params: ParamValues, r: np.ndarray
) -> np.ndarray:
    """
    Return the azimuthal mean of each density polar trait of a disk where
    its selection traits keep it, at the radii r (arcsec, on the disk):
    an array of shape (traits, len(r)).
    """
    r = r[:, None]

    def values(theta: np.ndarray) -> np.ndarray:
        out = np.asarray(
            plan.evaluate_polar_traits(params, 'rpt', r, theta), dtype=float)
        return out * _selected(disk, plan, params, r, theta)
    kinds = ('rpt', 'spt')
    return _azimuthal_mean(
        values, _axisymmetric(disk, kinds),
        _edges(disk, params, kinds, r[:, 0]))


def surface_density(
        disk: NodalDisk, plan: DiskPlan, params: ParamValues, r: np.ndarray
) -> np.ndarray:
    """
    Return the azimuthal mean of the surface density of a disk (the sum of
    its density polar traits, where its selection traits keep it) at the
    radii r (arcsec, on the disk).
    """
    return _surface_densities(
        disk, plan, _with_inputs(disk, params),
        np.asarray(r, dtype=float)).sum(axis=0)


@dataclasses.dataclass(frozen=True)
class _VerticalProfile:
    """
    The vertical profile of a thick disk at given radii (arcsec): the
    heights above its midplane (z, arcsec) and their weights, and the
    density at the radii and the heights (density, shape (radii,
    heights)), sum_i S_i p_i(z), over its density polar traits (S_i,
    their azimuthal means where its selection traits keep it) and their
    density height traits (pdfs, as the disk evaluates them: truncated at
    their zmax), with their integrals over z (mass, of shape (traits,
    radii); 0 for a pdf of less than 0.001 of its profile within zmax).
    By Gauss-Legendre quadrature on panels of geometric widths up to the
    zmax of each height trait.
    """
    z: np.ndarray
    weights: np.ndarray
    surfaces: np.ndarray
    pdfs: np.ndarray

    @classmethod
    def of(
            cls,
            disk: NodalDisk,
            plan: DiskPlan,
            params: ParamValues,
            radii: np.ndarray
    ) -> '_VerticalProfile':
        """
        Return the profile of a disk at the radii (see rms_heights for the
        arguments).

        Raises
        ------
        ValueError
            If a height pdf does not integrate to 1 (within 1e-6) where
            its trait has density (see rms_heights).
        """
        params = _with_inputs(disk, params)
        r = np.asarray(radii, dtype=float)
        edges = np.unique(np.concatenate(
            [_HEIGHT_EDGES * trait.zmax() for trait in disk.traits('rht')]))
        x, w = np.polynomial.legendre.leggauss(_HEIGHT_NPOINTS)
        lo, hi = edges[:-1, None], edges[1:, None]
        z = ((hi - lo) / 2 * x + (hi + lo) / 2).ravel()
        weights = ((hi - lo) / 2 * w).ravel()
        pdfs = np.asarray(plan.evaluate_height_traits(
            params, 'rht', r[:, None], z[None, :]), dtype=float)
        surfaces = _surface_densities(disk, plan, params, r)
        profile = cls(z, weights, surfaces, pdfs)
        mass = profile.mass()
        unresolved = (np.abs(mass - 1) > _PDF_TOLERANCE) & (mass != 0) \
            & (surfaces != 0)
        if np.any(unresolved):
            worst = mass[unresolved][np.argmax(np.abs(mass[unresolved] - 1))]
            raise ValueError(
                f"a density height trait of a disk integrates to {worst} "
                f"over the heights of its vertical profile, not 1: its "
                f"heights are not resolved (e.g. a sharp cut below its "
                f"zmax)")
        return profile

    def mass(self) -> np.ndarray:
        """
        Return the integral of each pdf over z (both sides of the
        midplane): an array of shape (traits, radii).
        """
        return 2 * self.pdfs @ self.weights

    def density(self) -> np.ndarray:
        """
        Return sum_i S_i p_i(z): an array of shape (radii, heights).
        """
        return np.einsum('tr,trz->rz', self.surfaces, self.pdfs)


def rms_heights(
        disk: NodalDisk,
        plan: DiskPlan,
        params: ParamValues,
        radii: np.ndarray
) -> np.ndarray:
    """
    Return the rms height sqrt(<z^2>) of the density of a thick disk about
    its midplane, at radii (arcsec, on the disk), in arcsec: <z^2> = sum_i
    S_i <z^2>_i / sum_i S_i, over its density polar traits (S_i, their
    azimuthal means where its selection traits keep it) and their density
    height traits (pdfs, as the disk evaluates them: truncated at their
    zmax). 0 where it has no density. The moments by Gauss-Legendre
    quadrature on panels of geometric widths up to the zmax of each
    height trait.

    Parameters
    ----------
    disk : NodalDisk
        The disk.
    plan : DiskPlan
        A plan of the disk, which evaluates its traits.
    params : ParamValues
        The values of the parameters of the disk (see NodalDiskDensity).
    radii : np.ndarray
        The radii (arcsec).

    Raises
    ------
    ValueError
        If a height pdf does not integrate to 1 (within 1e-6) where its
        trait has density: its heights are not resolved (e.g. a sharp
        cut, such as that of a uniform height trait, below its zmax).
    """
    profile = _VerticalProfile.of(disk, plan, params, radii)
    # (both sides of the midplane)
    total = np.sum(profile.surfaces * profile.mass(), axis=0)
    moment = 2 * profile.density() @ (profile.weights * profile.z ** 2)
    return np.sqrt(np.divide(
        moment, total, out=np.zeros_like(total), where=total > 0))


def jeans_dispersions(
        disk: NodalDisk,
        plan: DiskPlan,
        params: ParamValues,
        radii: np.ndarray,
        vertical_force: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    Return the dispersion (km/s) of the equilibrium 'galaxy' of a thick
    disk (see traits.DPTraitEquilibrium) at radii (arcsec, on the disk):
    sigma^2 = int_0^inf z rho(z) (-F_z(R, z)) dz / int_0^inf rho(z) dz,
    the dispersion of the column weighted by its density, with its
    density of rms_heights (on its heights) and the vertical force F_z of
    the mass of its galaxy. 0 where it has no density.

    Parameters
    ----------
    disk : NodalDisk
        The disk.
    plan : DiskPlan
        A plan of the disk, which evaluates its traits.
    params : ParamValues
        The values of the parameters of the disk (see NodalDiskDensity).
    radii : np.ndarray
        The radii (arcsec).
    vertical_force : callable
        The vertical force ((km/s)^2/arcsec) at the radii R and the
        heights z (arcsec) of two arrays, broadcast together.

    Raises
    ------
    ValueError
        If a height pdf does not integrate to 1 (see rms_heights).
    """
    profile = _VerticalProfile.of(disk, plan, params, radii)
    r = np.asarray(radii, dtype=float)[:, None]
    z = profile.z[None, :]
    density = profile.density()
    column = density @ profile.weights
    weight = (z * density * -vertical_force(r, z)) @ profile.weights
    return np.sqrt(np.divide(
        weight, column, out=np.zeros_like(column), where=column > 0))


def rotation(
        disk: NodalDisk,
        plan: DiskPlan,
        params: ParamValues,
        radii: np.ndarray,
        state: 'DiskState | None' = None
) -> np.ndarray:
    """
    Return the rotation of a disk at its midplane, at radii (arcsec, on
    the disk), in km/s: the azimuthal mean of the sum of its tangential
    velocity traits (their speeds), each times its velocity height trait
    at z = 0; 0 without them. With the pressure support of its tangential
    traits in its state (see DiskState.pressure), the mean v corrected
    for it, as the kernels draw it: sign(v) sqrt(max(0, v^2 + P)), P
    linear between its subnodes (without its state, its circular
    velocity).

    Parameters
    ----------
    disk : NodalDisk
        The disk.
    plan : DiskPlan
        A plan of the disk, which evaluates its traits.
    params : ParamValues
        The values of the parameters of the disk (see NodalDiskDensity).
    radii : np.ndarray
        The radii (arcsec).
    state : DiskState, optional
        What its galaxy gives it (e.g. the rotation of its velocity trait
        'mass', or the pressure support of its tangential traits), if
        anything.
    """
    params = _with_inputs(disk, params, state)
    r = np.asarray(radii, dtype=float)[:, None]
    tangential = [
        trait.direction() == 'tan' for trait in disk.traits('vpt')]
    if not any(tangential):
        return np.zeros(len(r))
    heights = _at_midplane(disk, plan, params, 'vht', r)

    def values(theta: np.ndarray) -> np.ndarray:
        speeds = plan.evaluate_polar_traits(params, 'vpt', r, theta) \
            * heights
        return np.sum(
            speeds[tangential], axis=0, keepdims=True, dtype=float)
    # (the other traits, e.g. a spiral wave, do not make its sum vary)
    axisymmetric = all(
        trait.axisymmetric()
        for trait, along in zip(disk.traits('vpt'), tangential) if along)
    v = _azimuthal_mean(
        values, axisymmetric, _edges(disk, params, ('vpt',), r[:, 0]))[0]
    if state is None or state.pressure is None:
        return v
    pressure = np.interp(r[:, 0], disk.subrnodes(), state.pressure)
    return np.copysign(np.sqrt(np.maximum(v ** 2 + pressure, 0)), v)


def density_waves(
        disk: NodalDisk,
        plan: DiskPlan,
        params: ParamValues,
        state: 'DiskState'
) -> ParamValues:
    """
    Return the tables of the density waves of a disk at its subnodes, by
    the names of their parameters (see traits.Spiral): in its
    rotation there (see rotation), with the derivative of the rotation by
    the second-order three-point formula, and for the density of a wave,
    in the density of the disk without its waves (the azimuthal mean of
    the sum of its other density polar traits).

    Parameters
    ----------
    disk : NodalDisk
        The disk.
    plan : DiskPlan
        A plan of the disk, which evaluates its traits.
    params : ParamValues
        The values of the parameters of the disk (see NodalDiskDensity).
    state : DiskState
        What its galaxy gives it so far: the rotation of its velocity
        traits 'mass', or the pressure support of its tangential traits,
        if any.
    """
    params = _with_inputs(disk, params, state)
    r = np.asarray(disk.subrnodes(), dtype=float)
    v = rotation(disk, plan, params, r, state)
    dv = np.gradient(v, r, edge_order=2)
    density = _density_without_waves(disk, plan, params, r) \
        if any(isinstance(trait, RPTraitSpiral)
               for trait in disk.traits('rpt')) else None
    tables = {}
    for kind in ('rpt', 'vpt'):
        for trait, values, names in zip(
                disk.traits(kind), disk.trait_values(kind, params),
                disk.trait_params(kind).pnames):
            if isinstance(trait, (RPTraitSpiral, VPTraitSpiral)):
                tables |= {
                    names[name]: table for name, table in trait.wave(
                        r, v, dv, density, values).items()}
    return tables


def _density_without_waves(
        disk: NodalDisk, plan: DiskPlan, params: ParamValues, r: np.ndarray
) -> np.ndarray:
    """
    Return the azimuthal mean of the sum of the density polar traits of a
    disk but its density waves, at the radii r (arcsec, on the disk), of
    all of the disk (not only where its selection traits keep it).
    """
    others = [
        not isinstance(trait, RPTraitSpiral) for trait in disk.traits('rpt')]
    r = r[:, None]

    def values(theta: np.ndarray) -> np.ndarray:
        densities = plan.evaluate_polar_traits(params, 'rpt', r, theta)
        return np.sum(
            densities[others], axis=0, keepdims=True, dtype=float)
    axisymmetric = all(
        trait.axisymmetric()
        for trait, other in zip(disk.traits('rpt'), others) if other)
    return _azimuthal_mean(
        values, axisymmetric, _edges(disk, params, ('rpt',), r[:, 0]))[0]


def dispersion(
        disk: NodalDisk,
        plan: DiskPlan,
        params: ParamValues,
        radii: np.ndarray,
        equilibrium: np.ndarray | None = None,
        direction: str | None = None
) -> np.ndarray:
    """
    Return the velocity dispersion of a disk at its midplane, at radii
    (arcsec, on the disk), in km/s: the azimuthal mean of the sum of its
    dispersion polar traits times its dispersion height traits at z = 0,
    with the dispersion of its vertical equilibrium in quadrature, if
    any (as pressure_support takes it); along a direction, if given
    ('rad', 'tan' or 'ver': its isotropic traits and those along it, see
    traits), as it must be for a disk whose dispersion is not isotropic.

    Parameters
    ----------
    disk : NodalDisk
        The disk.
    plan : DiskPlan
        A plan of the disk, which evaluates its traits.
    params : ParamValues
        The values of the parameters of the disk (see NodalDiskDensity).
    radii : np.ndarray
        The radii (arcsec).
    equilibrium : np.ndarray, optional
        The dispersion of the vertical equilibrium of the disk at its
        subnodes (km/s, see traits.DPTraitEquilibrium), interpolated linearly
        between them, as the disk is evaluated.
    direction : str, optional
        The direction: 'rad', 'tan' or 'ver'.

    Raises
    ------
    ValueError
        If the direction is not valid, or not given for a disk whose
        dispersion is not isotropic.
    """
    if direction is None and not _isotropic(disk):
        raise ValueError(
            "the dispersion of the disk is not isotropic: give its "
            "direction, 'rad', 'tan' or 'ver'")
    if direction not in (None, 'rad', 'tan', 'ver'):
        raise ValueError(
            f"direction must be 'rad', 'tan' or 'ver'; it is {direction!r}")
    params = _with_inputs(disk, params)
    r = np.asarray(radii, dtype=float)[:, None]
    balanced = np.zeros(r.shape) if equilibrium is None \
        else np.interp(r, disk.subrnodes(), equilibrium)
    if not disk.traits('dpt'):
        return balanced[:, 0]
    heights = _at_midplane(disk, plan, params, 'dht', r)

    def values(theta: np.ndarray) -> np.ndarray:
        return _dispersions(
            disk,
            plan.evaluate_polar_traits(params, 'dpt', r, theta) * heights,
            balanced, (direction or DPT_DIRECTION_DEFAULT,))[0][None]
    kinds = ('dpt',)
    return _azimuthal_mean(
        values, _axisymmetric(disk, kinds),
        _edges(disk, params, kinds, r[:, 0]))[0]


def pressure_support(
        disk: NodalDisk,
        plan: DiskPlan,
        params: ParamValues,
        radii: np.ndarray,
        density: np.ndarray | None = None,
        equilibrium: np.ndarray | None = None,
        heights: np.ndarray | None = None,
        midplane: np.ndarray | None = None
) -> np.ndarray:
    """
    Return the pressure support of a disk, (R / rho) d(rho sigma^2)/dR
    ((km/s)^2), at radii (arcsec, increasing, on the disk), at its
    midplane or at given heights above it (the radial Jeans equation, see
    traits.PressureSupportJeans and traits.VHTraitMass), with the
    azimuthal means of its density rho at each height (its density polar
    traits times its density height traits, where its selection traits
    keep it; of a thin disk, its density polar traits, times its density
    at its midplane per unit surface density, if given, as the given
    density is) and of rho
    sigma^2, sigma its velocity dispersion at each height (its dispersion
    polar traits times its dispersion height traits, and the dispersion
    of its vertical equilibrium in quadrature, if any). The derivative by
    the second-order three-point formula on the radii. NaN where it has
    no density.

    A disk whose dispersion is not isotropic (see traits) has the radial
    Jeans equation of its velocity ellipsoid: (R / rho) d(rho
    sigma_R^2)/dR + sigma_R^2 - sigma_phi^2, with the azimuthal means of
    rho sigma_R^2 and of rho sigma_phi^2 (the means of sigma_R^2 -
    sigma_phi^2 weighted by rho). With every dispersion isotropic, the
    support above.

    Parameters
    ----------
    disk : NodalDisk
        The disk.
    plan : DiskPlan
        A plan of the disk, which evaluates its traits.
    params : ParamValues
        The values of the parameters of the disk (see NodalDiskDensity).
    radii : np.ndarray
        The radii (arcsec).
    density : np.ndarray, optional
        The density (e.g. of its fluid) to take instead of that of the
        disk: at the radii at the midplane (in any unit), or at the radii
        and the heights (an array of shape (len(radii), len(heights))); of
        a thin disk, the surface density.
    equilibrium : np.ndarray, optional
        The dispersion of the vertical equilibrium of the disk at its
        subnodes (km/s, see traits.DPTraitEquilibrium), interpolated
        linearly between them, as the disk is evaluated.
    heights : np.ndarray, optional
        The heights (arcsec) above the midplane; by default, the midplane.
    midplane : np.ndarray, optional
        Of a thin disk that does not draw its vertical equilibrium, its
        density at its midplane per unit surface density at the radii (1 /
        arcsec, see traits.RHTraitEquilibrium.midplane_density).

    Returns
    -------
    np.ndarray
        The pressure support at the radii (an array of their shape), or
        at the radii and the heights (of shape (len(radii),
        len(heights))).
    """
    params = _with_inputs(disk, params)
    r = np.asarray(radii, dtype=float)[:, None]
    z = np.zeros(1) if heights is None else np.asarray(heights, dtype=float)
    # (the height traits before the azimuths: of shape (traits, heights,
    # radii, 1))
    density_heights = _at_heights(disk, plan, params, 'rht', r, z)
    dispersion_heights = _at_heights(disk, plan, params, 'dht', r, z)
    given = None if density is None \
        else np.asarray(density, dtype=float).reshape(len(r), -1).T[..., None]
    if midplane is not None:
        factor = np.asarray(midplane, dtype=float)[:, None]
        density_heights = density_heights * factor
        given = None if given is None else given * factor
    balanced = 0 if equilibrium is None \
        else np.interp(r, disk.subrnodes(), equilibrium)
    isotropic = _isotropic(disk)
    directions = ('iso',) if isotropic else ('rad', 'tan', 'ver')

    def values(theta: np.ndarray) -> np.ndarray:
        # (sigma_R, sigma_phi and sigma_z, the same if isotropic)
        sigmas = _dispersions(
            disk,
            plan.evaluate_polar_traits(params, 'dpt', r, theta)[:, None]
            * dispersion_heights, balanced, directions)
        if isotropic:
            radial = tangential = sigmas[0]
        else:
            radial, tangential, _ = sigmas
        if given is not None:
            own = np.broadcast_to(given, np.broadcast_shapes(
                given.shape, radial.shape))
        else:
            own = np.sum(
                plan.evaluate_polar_traits(params, 'rpt', r, theta)[:, None]
                * density_heights, axis=0, dtype=float)
            own = own * _selected(disk, plan, params, r, theta)
        rows = (own, own * radial ** 2)
        if not isotropic:
            rows += (own * tangential ** 2,)
        rows = np.stack(rows)
        # (the heights in the rows of the azimuthal means)
        return rows.reshape(-1, *rows.shape[2:])
    kinds = ('rpt', 'spt', 'dpt') if density is None else ('dpt',)
    density, pressure, *tangential = _azimuthal_mean(
        values, _axisymmetric(disk, kinds),
        _edges(disk, params, kinds, r[:, 0])).reshape(-1, len(z), len(r))
    gradient = np.gradient(pressure, r[:, 0], axis=-1, edge_order=2)
    support = r[:, 0] * gradient
    if not isotropic:
        # (rho (sigma_R^2 - sigma_phi^2))
        support = support + (pressure - tangential[0])
    support = np.divide(
        support, density, out=np.full_like(density, np.nan),
        where=density != 0).T
    return support[:, 0] if heights is None else support


class NodalDiskDensity:
    """
    The density of a disk for the mass components that follow it (see
    mass.DiskDensity): the azimuthal means of its density polar
    traits, where its selection traits keep it, and its density height
    traits, as it evaluates them with the given values of its
    parameters, in kpc. Its rings are taken flat (coplanar and
    concentric).

    Parameters
    ----------
    disk : NodalDisk
        The disk.
    plan : DiskPlan
        A plan of the disk, which evaluates its traits.
    params : ParamValues
        The values of the parameters of the disk (see NodalDisk.pdescs): those
        that the model gives it (see NodalDisk.circular_velocity_params) can be
        left out, since its density does not depend on them.
    kpc_per_arcsec : float
        The length of an arcsec in the disk.
    mass_per_unit : float, optional
        The mass surface density (Msun/kpc^2) of one unit of the density,
        if it has a physical unit (see _units.mass_per_unit).
    """

    def __init__(
            self,
            disk: NodalDisk,
            plan: DiskPlan,
            params: ParamValues,
            kpc_per_arcsec: float,
            mass_per_unit: float | None = None
    ):
        self._disk = disk
        self._plan = plan
        self._params = _with_inputs(disk, params)
        self._kpc_per_arcsec = kpc_per_arcsec
        self._mass_per_unit = mass_per_unit

    def knots(self, rstep: float | None) -> np.ndarray:
        """
        Return the radii (kpc) of its subnodes, or of the subnodes of rings
        of width at most rstep (arcsec, see _nodal.subring_radii), and
        beyond the last, of its tail, to its end (see _nodal.tail_radii
        and tail_end).
        """
        radii = self._disk.subrnodes() if rstep is None \
            else subring_radii(self._disk.rnodes(), rstep)
        return np.array(
            radii + tail_radii(radii) + (tail_end(radii),)) \
            * self._kpc_per_arcsec

    def thick(self) -> bool:
        """Check whether it has density height traits."""
        return bool(self._disk.traits('rht'))

    def mass_per_unit(self) -> float | None:
        """
        Return the mass surface density (Msun/kpc^2) of one unit of the
        density, if it has a physical unit; None without one.
        """
        return self._mass_per_unit

    def surface_densities(self, R: np.ndarray) -> np.ndarray:
        """
        Return the azimuthal mean of each density polar trait (in the units
        of the density) where the selection traits keep the disk, at the
        radii R (kpc, on the disk): an array of shape (traits, len(R));
        0 from the end of its tail on (see knots), to which it goes on
        linearly from the knot before (a thin disk would pull infinitely at
        a jump there; the tail is below exp(-10) of its value at the last
        ring, see _nodal.tail_end).
        """
        r = self._radii(R)
        end = tail_end(self._disk.subrnodes())
        return np.where(
            r < end,
            _surface_densities(self._disk, self._plan, self._params, r), 0)

    def height_pdfs(self, R: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Return the pdf of the height (1/kpc) of each density polar trait,
        its density height trait, at the radii R (kpc, on the disk) and the
        heights z (kpc): an array of shape (traits, len(R), len(z)).
        """
        pdfs = self._plan.evaluate_height_traits(
            self._params, 'rht', self._radii(R)[:, None],
            np.asarray(z)[None, :] / self._kpc_per_arcsec)
        return np.asarray(pdfs, dtype=float) / self._kpc_per_arcsec

    def _radii(self, R: np.ndarray) -> np.ndarray:
        """Return the radii R (kpc, on the disk) in arcsec."""
        return np.asarray(R) / self._kpc_per_arcsec
