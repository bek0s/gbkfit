"""
The radial profiles of a disk, as its galaxy takes them for gravity: the
azimuthal means of its traits, for the mass components that follow it
(see mass.ComponentDensity) and for its pressure support (see
traits.VPTraitMass).
"""

from collections.abc import Callable, Sequence

import numpy as np

from gbkfit.params import ParamValues
from ._disk import Disk, DiskPlan, subring_radii


__all__ = [
    'DiskDensity',
    'pressure_support'
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


def pressure_support(
        disk: Disk,
        plan: DiskPlan,
        params: ParamValues,
        radii: np.ndarray,
        mode: str,
        density: np.ndarray | None = None
) -> np.ndarray:
    """
    Return the pressure support of a disk at its midplane, (R / rho)
    d(rho sigma^2)/dR ((km/s)^2), at radii (arcsec, increasing, on the
    disk), with sigma its velocity dispersion at the midplane (its
    dispersion polar traits times its dispersion height traits at z = 0),
    by mode (see traits.VPTraitMass): 'jeans', with the azimuthal means
    of its density rho at the midplane (its density polar traits times its
    density height traits at z = 0, or its density polar traits if thin,
    where its selection traits keep it) and of rho sigma^2;
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
    """
    params = _with_given(disk, params)
    r = np.asarray(radii, dtype=float)[:, None]

    def at_midplane(kind: str) -> np.ndarray:
        # The height traits of a kind at z = 0, which multiply the polar
        # traits they pair with (1 without them)
        if not disk.traits(kind):
            return np.ones((1, len(r), 1))
        return plan.evaluate_height_traits(params, kind, r, np.zeros(1))

    density_heights = at_midplane('rht') if mode == 'jeans' \
        else np.ones((1, len(r), 1))
    dispersion_heights = at_midplane('dht')
    given = None if density is None \
        else np.asarray(density, dtype=float)[:, None]

    def values(theta: np.ndarray) -> np.ndarray:
        dispersion = np.sum(
            plan.evaluate_polar_traits(params, 'dpt', r, theta)
            * dispersion_heights, axis=0, dtype=float)
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
        of width at most rstep (arcsec, see _disk.subring_radii).
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
        r = self._radii(R)[:, None]

        def values(theta: np.ndarray) -> np.ndarray:
            out = np.asarray(self._plan.evaluate_polar_traits(
                self._params, 'rpt', r, theta), dtype=float)
            return out * _selected(
                self._disk, self._plan, self._params, r, theta)
        kinds = ('rpt', 'spt')
        return _azimuthal_mean(
            values, _axisymmetric(self._disk, kinds),
            _edges(self._disk, self._params, kinds, r[:, 0]))

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
