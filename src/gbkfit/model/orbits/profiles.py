"""
Spherical profiles of the light of galaxies: the tracers whose orbits
Schwarzschild models superpose (lengths in kpc, light in Lsun).
"""

import dataclasses

import numpy as np
import numpy.typing as npt

from gbkfit.model.mass import MassModel, MassPoint, MassSpheroid, ProfileZhao
from gbkfit.params import ParamValues


__all__ = [
    'Hernquist'
]


@dataclasses.dataclass(frozen=True)
class Hernquist:
    """
    The profile of Hernquist (1990), of total light (or mass) `total` and
    scale radius `a` (kpc): the density total a / (2 pi r (r + a)^3).
    """
    total: float
    a: float

    def density(self, r: npt.ArrayLike) -> np.ndarray:
        """Return its density (per kpc^3) at the spherical radii r."""
        r = np.asarray(r, dtype=float)
        return self.total * self.a / (2 * np.pi * r * (r + self.a) ** 3)

    def enclosed(self, r: npt.ArrayLike) -> np.ndarray:
        """Return its light within the spherical radii r."""
        r = np.asarray(r, dtype=float)
        return self.total * r ** 2 / (r + self.a) ** 2

    def surface_density(self, R: npt.ArrayLike) -> np.ndarray:
        """
        Return its surface density (per kpc^2) at the projected radii R
        (Hernquist 1990, eqs. 32-33).
        """
        s = np.asarray(R, dtype=float) / self.a
        # X(s): arcsech(s) / sqrt(1 - s^2) inside s = 1, arcsec(s) /
        # sqrt(s^2 - 1) outside it
        with np.errstate(divide='ignore', invalid='ignore'):
            inside = np.arccosh(1 / s) / np.sqrt(1 - s ** 2)
            outside = np.arccos(1 / s) / np.sqrt(s ** 2 - 1)
            x = np.where(s < 1, inside, outside)
            sigma = ((2 + s ** 2) * x - 3) / (1 - s ** 2) ** 2
        # (near s = 1, where the terms cancel, its Taylor series)
        e = s - 1
        sigma = np.where(
            np.abs(e) < 1e-3, 4 / 15 - 16 / 35 * e + 8 / 15 * e ** 2, sigma)
        return self.total / (2 * np.pi * self.a ** 2) * sigma

    def mass_model(
            self, upsilon: float, bh_mass: float
    ) -> tuple[MassModel, ParamValues]:
        """
        Return the mass model of stars of this light and of the mass-to-
        light ratio upsilon (Msun/Lsun), with a central black hole of
        bh_mass (Msun), and the values of its parameters: of the
        components 'stars' (the Zhao profile of (alpha, beta, gamma) = (1,
        4, 1), which is Hernquist's) and 'bh' (a point mass).
        """
        model = MassModel([
            MassSpheroid('stars', ProfileZhao(parameters='density_radius')),
            MassPoint('bh')])
        with np.errstate(divide='ignore'):
            params = dict(
                mass_stars_log_rho=np.log10(
                    upsilon * self.total / (2 * np.pi * self.a ** 3)),
                mass_stars_rs=self.a, mass_stars_alpha=1.0,
                mass_stars_beta=4.0, mass_stars_gamma=1.0,
                mass_bh_log_m=np.log10(bh_mass))
        return model, params
