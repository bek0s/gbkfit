"""
The radial profiles of a disk, as its galaxy takes them for gravity: the
azimuthal means of its traits, for the mass components that follow it
(see mass.ComponentDensity) and for its pressure support (see
traits.VPTraitMass).
"""

from collections.abc import Callable

import numpy as np

from gbkfit.params import ParamValues
from ._disk import Disk, DiskPlan, subring_radii


__all__ = [
    'DiskDensity',
    'pressure_support'
]


# The azimuthal means: the trapezoid rule on at least and at most these
# numbers of azimuths, doubled until two means agree to the tolerance,
# relative to the largest value
_NAZIMUTHS_MIN = 64
_NAZIMUTHS_MAX = 4096
_AZIMUTHAL_TOLERANCE = 1e-6


def _azimuthal_mean(values: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Return the mean over the azimuth of values(theta), an array of shape
    (rows, radii, len(theta)) at the azimuths theta (degrees), as an array
    of shape (rows, radii): by the trapezoid rule, exact for harmonics of
    orders below the number of azimuths, to _AZIMUTHAL_TOLERANCE of the
    largest absolute value of each row. Structure narrower than the
    spacing of the azimuths (2 pi R / 128 at least) can go unseen by two
    successive numbers of azimuths, and be missed.

    Raises
    ------
    ValueError
        If the mean does not converge on _NAZIMUTHS_MAX azimuths.
    """
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
        radii: np.ndarray
) -> np.ndarray:
    """
    Return the pressure support of a disk at its midplane, (R / rho)
    d(rho sigma^2)/dR ((km/s)^2), at radii (arcsec, increasing, on the
    disk): with the azimuthal means of its density rho at the midplane
    (its density polar traits times its density height traits at z = 0,
    or its density polar traits if thin, where its selection traits keep
    it)
    and of rho sigma^2, with sigma its velocity dispersion at the midplane
    (its dispersion polar traits times its dispersion height traits at z =
    0); the derivative by the second-order three-point formula on the
    radii (see traits.VPTraitMass). NaN where it has no density.

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

    def at_midplane(kind: str) -> np.ndarray:
        # The height traits of a kind at z = 0, which multiply the polar
        # traits they pair with (1 without them)
        if not disk.traits(kind):
            return np.ones((1, len(r), 1))
        return plan.evaluate_height_traits(params, kind, r, np.zeros(1))

    density_heights = at_midplane('rht')
    dispersion_heights = at_midplane('dht')

    def values(theta: np.ndarray) -> np.ndarray:
        density = np.sum(
            plan.evaluate_polar_traits(params, 'rpt', r, theta)
            * density_heights, axis=0, dtype=float)
        density *= _selected(disk, plan, params, r, theta)
        dispersion = np.sum(
            plan.evaluate_polar_traits(params, 'dpt', r, theta)
            * dispersion_heights, axis=0, dtype=float)
        return np.stack((density, density * dispersion ** 2))
    density, pressure = _azimuthal_mean(values)
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
    """

    def __init__(
            self,
            disk: Disk,
            plan: DiskPlan,
            params: ParamValues,
            kpc_per_arcsec: float
    ):
        self._disk = disk
        self._plan = plan
        self._params = _with_given(disk, params)
        self._kpc_per_arcsec = kpc_per_arcsec

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
        return _azimuthal_mean(values)

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
