"""
The radial profiles of a disk, as the mass components of its model take
them: the azimuthal means of its traits (see mass.ComponentDensity).
"""

from collections.abc import Callable

import numpy as np

from gbkfit.params import ParamValues
from ._disk import Disk, DiskPlan, subring_radii


__all__ = [
    'DiskDensity'
]


# The azimuthal means: the trapezoid rule on at least and at most these
# numbers of azimuths, doubled until two means agree to the tolerance,
# relative to the largest
_NAZIMUTHS_MIN = 64
_NAZIMUTHS_MAX = 4096
_AZIMUTHAL_TOLERANCE = 1e-6


def _azimuthal_mean(values: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Return the mean over the azimuth of values(theta), an array whose last
    axis is that of the azimuths theta (degrees): by the trapezoid rule,
    exact for harmonics of orders below the number of azimuths.

    Raises
    ------
    ValueError
        If the mean does not converge on _NAZIMUTHS_MAX azimuths.
    """
    count = _NAZIMUTHS_MIN
    mean = values(np.arange(count) * 360 / count).mean(axis=-1)
    while count < _NAZIMUTHS_MAX:
        between = values((np.arange(count) + 0.5) * 360 / count)
        refined = (mean + between.mean(axis=-1)) / 2
        change = np.max(np.abs(refined - mean), initial=0)
        mean = refined
        count *= 2
        if change <= _AZIMUTHAL_TOLERANCE * np.max(np.abs(mean), initial=0):
            return mean
    raise ValueError(
        f"the azimuthal mean of the brightness of a disk does not converge "
        f"on {_NAZIMUTHS_MAX} azimuths: its azimuthal structure is too fine "
        f"or too sharp (e.g. narrow blobs, or the edges of selection "
        f"traits)")


class DiskDensity:
    """
    The density of a disk for the mass components that follow it (see
    mass.ComponentDensity): the azimuthal means of its density polar
    traits (its brightness), where its selection traits keep it, and its
    density height traits, as it evaluates them with the given values of
    its parameters, in kpc. Its rings are taken flat (coplanar and
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
        # (the values that the model gives do not change the density)
        given = {
            name: np.full(len(radii), np.nan)
            for name, radii in disk.circular_velocity_params().items()}
        self._disk = disk
        self._plan = plan
        self._params = given | dict(params)
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
        of its brightness) where the selection traits keep the disk, at the
        radii R (kpc, on the disk): an array of shape (traits, len(R)).
        """
        r = self._radii(R)[:, None]
        selected = bool(self._disk.traits('spt'))

        def values(theta: np.ndarray) -> np.ndarray:
            out = np.asarray(self._plan.evaluate_polar_traits(
                self._params, 'rpt', r, theta), dtype=float)
            if selected:
                out *= self._plan.evaluate_polar_traits(
                    self._params, 'spt', r, theta).sum(axis=0) != 0
            return out
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
