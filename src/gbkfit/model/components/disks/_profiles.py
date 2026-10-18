"""
The radial profiles of a disk, as its galaxy takes them for gravity: the
azimuthal means of its traits, for the mass components that follow it
(see mass.ComponentDensity), for its pressure support (see
traits.VPTraitMass), and its rms height, rotation and jeans dispersion,
for its vertical equilibrium (see parts.Kinematics); and its rotation
and dispersion, for the functions of its galaxy (see GalaxyFunctions).
"""

import dataclasses
from collections.abc import Callable, Sequence

import numpy as np

from gbkfit.params import ParamValues
from .._nodal import subring_radii
from ._disk import Disk, DiskPlan


__all__ = [
    'DiskDensity',
    'dispersion',
    'jeans_dispersions',
    'pressure_support',
    'rms_heights',
    'rotation'
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
        disk: Disk, params: ParamValues, kinds: Sequence[str], r: np.ndarray
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


def _axisymmetric(disk: Disk, kinds: Sequence[str]) -> bool:
    """
    Check whether the polar traits of the given kinds of a disk all have
    the same value at every azimuth (see RPTrait.axisymmetric).
    """
    return all(
        trait.axisymmetric() for kind in kinds for trait in disk.traits(kind))


def _with_given(disk: Disk, params: ParamValues) -> ParamValues:
    """
    Return the values of the parameters of a disk with those that its
    model gives it (see Disk.circular_velocity_params) as NaN, if left
    out: its profiles do not depend on them.
    """
    given = {
        name: np.full(len(radii), np.nan)
        for name, radii in disk.circular_velocity_params().items()}
    return given | dict(params)


def _selected(
        disk: Disk, plan: DiskPlan, params: ParamValues, r: np.ndarray,
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
        disk: Disk, plan: DiskPlan, params: ParamValues, kind: str,
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


def _surface_densities(
        disk: Disk, plan: DiskPlan, params: ParamValues, r: np.ndarray
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
            disk: Disk,
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
        params = _with_given(disk, params)
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
        disk: Disk,
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
    disk : Disk
        The disk.
    plan : DiskPlan
        A plan of the disk, which evaluates its traits.
    params : ParamValues
        The values of the parameters of the disk (see DiskDensity).
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
        disk: Disk,
        plan: DiskPlan,
        params: ParamValues,
        radii: np.ndarray,
        vertical_force: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    Return the dispersion (km/s) of the jeans equilibrium of a thick disk
    (see parts.Kinematics) at radii (arcsec, on the disk): sigma^2 = int_0^inf
    rho(z) (-F_z(R, z)) dz / rho(0), the weight of the column of the disk
    above its midplane over its density there, with its density of
    rms_heights (on its heights) and the vertical force F_z of the mass
    of its galaxy. 0 where it has no density at its midplane.

    Parameters
    ----------
    disk : Disk
        The disk.
    plan : DiskPlan
        A plan of the disk, which evaluates its traits.
    params : ParamValues
        The values of the parameters of the disk (see DiskDensity).
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
    midplane = np.sum(
        profile.surfaces * _at_midplane(
            disk, plan, _with_given(disk, params), 'rht', r)[..., 0], axis=0)
    weight = (profile.density() * -vertical_force(r, profile.z[None, :])) \
        @ profile.weights
    return np.sqrt(np.divide(
        weight, midplane, out=np.zeros_like(midplane), where=midplane > 0))


def rotation(
        disk: Disk,
        plan: DiskPlan,
        params: ParamValues,
        radii: np.ndarray
) -> np.ndarray:
    """
    Return the rotation of a disk at its midplane, at radii (arcsec, on
    the disk), in km/s: the azimuthal mean of the sum of its tangential
    velocity traits (their speeds), each times its velocity height trait
    at z = 0; 0 without them.

    Parameters
    ----------
    disk : Disk
        The disk.
    plan : DiskPlan
        A plan of the disk, which evaluates its traits.
    params : ParamValues
        The values of the parameters of the disk (see DiskDensity).
    radii : np.ndarray
        The radii (arcsec).
    """
    params = _with_given(disk, params)
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
    kinds = ('vpt',)
    return _azimuthal_mean(
        values, _axisymmetric(disk, kinds),
        _edges(disk, params, kinds, r[:, 0]))[0]


def dispersion(
        disk: Disk,
        plan: DiskPlan,
        params: ParamValues,
        radii: np.ndarray,
        equilibrium: np.ndarray | None = None
) -> np.ndarray:
    """
    Return the velocity dispersion of a disk at its midplane, at radii
    (arcsec, on the disk), in km/s: the azimuthal mean of the sum of its
    dispersion polar traits times its dispersion height traits at z = 0,
    with the dispersion of its vertical equilibrium in quadrature, if
    any (as pressure_support takes it).

    Parameters
    ----------
    disk : Disk
        The disk.
    plan : DiskPlan
        A plan of the disk, which evaluates its traits.
    params : ParamValues
        The values of the parameters of the disk (see DiskDensity).
    radii : np.ndarray
        The radii (arcsec).
    equilibrium : np.ndarray, optional
        The dispersion of the vertical equilibrium of the disk at its
        subnodes (km/s, see parts.Kinematics), interpolated linearly
        between them, as the disk is evaluated.
    """
    params = _with_given(disk, params)
    r = np.asarray(radii, dtype=float)[:, None]
    balanced = np.zeros(r.shape) if equilibrium is None \
        else np.interp(r, disk.subrnodes(), equilibrium)
    if not disk.traits('dpt'):
        return balanced[:, 0]
    heights = _at_midplane(disk, plan, params, 'dht', r)

    def values(theta: np.ndarray) -> np.ndarray:
        return np.hypot(np.sum(
            plan.evaluate_polar_traits(params, 'dpt', r, theta) * heights,
            axis=0, keepdims=True, dtype=float), balanced)
    kinds = ('dpt',)
    return _azimuthal_mean(
        values, _axisymmetric(disk, kinds),
        _edges(disk, params, kinds, r[:, 0]))[0]


def pressure_support(
        disk: Disk,
        plan: DiskPlan,
        params: ParamValues,
        radii: np.ndarray,
        mode: str,
        density: np.ndarray | None = None,
        equilibrium: np.ndarray | None = None
) -> np.ndarray:
    """
    Return the pressure support of a disk at its midplane, (R / rho)
    d(rho sigma^2)/dR ((km/s)^2), at radii (arcsec, increasing, on the
    disk), with sigma its velocity dispersion at the midplane (its
    dispersion polar traits times its dispersion height traits at z = 0,
    and the dispersion of its vertical equilibrium in quadrature, if
    any), by mode (see traits.VPTraitMass): 'jeans', with the azimuthal
    means of its density rho at the midplane (its density polar traits
    times its density height traits at z = 0, or its density polar traits
    if thin, where its selection traits keep it) and of rho sigma^2;
    'self_gravitating', with those of S^2 / sigma^2 and of S^2, which rho
    and rho sigma^2 are proportional to (S its density polar traits,
    where its selection traits keep it), or instead the given density.
    The derivative by the second-order three-point formula on the radii.
    NaN where it has no density.

    Parameters
    ----------
    disk : Disk
        The disk.
    plan : DiskPlan
        A plan of the disk, which evaluates its traits.
    params : ParamValues
        The values of the parameters of the disk (see DiskDensity).
    radii : np.ndarray
        The radii (arcsec).
    mode : str
        The pressure support: 'jeans' or 'self_gravitating'.
    density : np.ndarray, optional
        The density at the radii (e.g. of a mass component, in any unit)
        to take instead of that of the disk: at the midplane for 'jeans',
        the surface density for 'self_gravitating'.
    equilibrium : np.ndarray, optional
        The dispersion of the vertical equilibrium of the disk at its
        subnodes (km/s, see parts.Kinematics), interpolated linearly
        between them, as the disk is evaluated.
    """
    params = _with_given(disk, params)
    r = np.asarray(radii, dtype=float)[:, None]
    density_heights = _at_midplane(disk, plan, params, 'rht', r) \
        if mode == 'jeans' else np.ones((1, len(r), 1))
    dispersion_heights = _at_midplane(disk, plan, params, 'dht', r)
    given = None if density is None \
        else np.asarray(density, dtype=float)[:, None]
    balanced = 0 if equilibrium is None \
        else np.interp(r, disk.subrnodes(), equilibrium)

    def values(theta: np.ndarray) -> np.ndarray:
        dispersion = np.hypot(np.sum(
            plan.evaluate_polar_traits(params, 'dpt', r, theta)
            * dispersion_heights, axis=0, dtype=float), balanced)
        if given is not None:
            own = np.broadcast_to(given, dispersion.shape)
        else:
            own = np.sum(
                plan.evaluate_polar_traits(params, 'rpt', r, theta)
                * density_heights, axis=0, dtype=float)
            own *= _selected(disk, plan, params, r, theta)
        if mode == 'jeans':
            return np.stack((own, own * dispersion ** 2))
        # (rho ~ S^2 / sigma^2: infinite without dispersion, where the
        # disk has no thickness and no pressure)
        squared = own ** 2
        return np.stack((np.divide(
            squared, dispersion ** 2, where=dispersion != 0,
            out=np.where(squared != 0, np.inf, 0.0)), squared))
    kinds = ('rpt', 'spt', 'dpt') if density is None else ('dpt',)
    density, pressure = _azimuthal_mean(
        values, _axisymmetric(disk, kinds),
        _edges(disk, params, kinds, r[:, 0]))
    gradient = np.gradient(pressure, r[:, 0], edge_order=2)
    return np.divide(
        r[:, 0] * gradient, density, out=np.full_like(density, np.nan),
        where=density != 0)


class DiskDensity:
    """
    The density of a disk for the mass components that follow it (see
    mass.ComponentDensity): the azimuthal means of its density polar
    traits, where its selection traits keep it, and its density height
    traits, as it evaluates them with the given values of its
    parameters, in kpc. Its rings are taken flat (coplanar and
    concentric).

    Parameters
    ----------
    disk : Disk
        The disk.
    plan : DiskPlan
        A plan of the disk, which evaluates its traits.
    params : ParamValues
        The values of the parameters of the disk (see Disk.pdescs): those
        that the model gives it (see Disk.circular_velocity_params) can be
        left out, since its density does not depend on them.
    kpc_per_arcsec : float
        The length of an arcsec in the disk.
    mass_per_unit : float, optional
        The mass surface density (Msun/kpc^2) of one unit of the density,
        if it has a physical unit (see _units.mass_per_unit).
    """

    def __init__(
            self,
            disk: Disk,
            plan: DiskPlan,
            params: ParamValues,
            kpc_per_arcsec: float,
            mass_per_unit: float | None = None
    ):
        self._disk = disk
        self._plan = plan
        self._params = _with_given(disk, params)
        self._kpc_per_arcsec = kpc_per_arcsec
        self._mass_per_unit = mass_per_unit

    def knots(self, rstep: float | None) -> np.ndarray:
        """
        Return the radii (kpc) of its subnodes, or of the subnodes of rings
        of width at most rstep (arcsec, see _nodal.subring_radii).
        """
        radii = self._disk.subrnodes() if rstep is None \
            else subring_radii(self._disk.rnodes(), rstep)
        return np.array(radii) * self._kpc_per_arcsec

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
        radii R (kpc, on the disk): an array of shape (traits, len(R)).
        """
        return _surface_densities(
            self._disk, self._plan, self._params, self._radii(R))

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
        """
        Return the radii R (kpc, on the disk) in arcsec, on the disk
        despite their rounding.
        """
        rnodes = self._disk.rnodes()
        return np.clip(
            np.asarray(R) / self._kpc_per_arcsec, rnodes[0], rnodes[-1])
