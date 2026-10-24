"""
The distance of a galaxy: the length of an arcsec at the galaxy, and the
epoch of the quantities of the galaxy that depend on the cosmology (e.g.
the overdensities that define the mass of a halo).

Inside, the distance is always the angular diameter distance. It is
given directly (as a parameter, in Mpc), or by the redshift of the
galaxy (see Galaxy; or a fixed one of its own) and a cosmology, which
give it. A redshift never follows from a given distance: for nearby
galaxies, peculiar velocities make it wrong.
"""

import abc
import functools
from typing import Any

import astropy.constants
import astropy.cosmology
import astropy.units
import numpy as np

from gbkfit.params import ParamValues
from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from .mass.units import G


__all__ = [
    'Distance',
    'DistanceDirect',
    'DistanceRedshift',
    'KPC_PER_ARCSEC_MPC',
    'distance_parser'
]


# The length (kpc) of one arcsec at an angular diameter distance of 1 Mpc
KPC_PER_ARCSEC_MPC = (1 * astropy.units.arcsec).to_value('rad') * 1e3

# The cosmology by default: Planck Collaboration (2020), Table 2
COSMOLOGY_DEFAULT = 'Planck18'

# The speed of light (km/s)
_C = astropy.constants.c.to_value('km/s')


def _cosmology(name: str) -> astropy.cosmology.Cosmology:
    """
    Return a cosmology of astropy by name (one of its realizations, e.g.
    'Planck18'). Raise ConfigError for an unknown name.
    """
    available = astropy.cosmology.realizations.available
    if name not in available:
        raise ConfigError(
            f"cosmology must be one of {list(available)}; it is {name!r}")
    return getattr(astropy.cosmology.realizations, name)


@functools.lru_cache(maxsize=256)
def _angular_diameter_distance(cosmology: str, redshift: float) -> float:
    """Return the angular diameter distance (Mpc) at a redshift."""
    distance = _cosmology(cosmology).angular_diameter_distance(redshift)
    return float(distance.to_value('Mpc'))


@functools.lru_cache(maxsize=256)
def _critical_density(cosmology: str, redshift: float) -> float:
    """Return the critical density (Msun/kpc^3) at a redshift."""
    density = _cosmology(cosmology).critical_density(redshift)
    return float(density.to_value('solMass / kpc3'))


@functools.lru_cache(maxsize=256)
def _omega_m(cosmology: str, redshift: float) -> float:
    """Return the matter density parameter at a redshift."""
    return float(_cosmology(cosmology).Om(redshift))


@functools.lru_cache(maxsize=256)
def _ricci_scalar(cosmology: str, redshift: float) -> float:
    """
    Return the Ricci scalar of the background universe (kpc^-2) at a
    redshift: 8 pi G (rho_m + (1 - 3 w) rho_de) / c^2, of its matter and
    its dark energy (radiation is traceless; the massive neutrinos, which
    astropy keeps apart from the matter, add some 0.05%: left out).
    """
    cosmo = _cosmology(cosmology)
    density = cosmo.critical_density(redshift).to_value('solMass / kpc3') * (
        cosmo.Om(redshift) + (1 - 3 * cosmo.w(redshift)) * cosmo.Ode(redshift))
    return float(8 * np.pi * G * density / _C ** 2)


def _baryon_fraction(cosmology: str) -> float | None:
    """
    Return the cosmic baryon fraction, Omega_b / Omega_m (None if the
    cosmology has no Omega_b).
    """
    cosmo = _cosmology(cosmology)
    return None if cosmo.Ob0 is None else float(cosmo.Ob0 / cosmo.Om0)


class Distance(parseutils.TypedSerializable, abc.ABC):
    """
    The distance of a galaxy: its parameters, the length of an arcsec at
    the galaxy, and the epoch and the cosmology of the quantities of the
    galaxy that depend on the cosmology.

    Parameters
    ----------
    cosmology : str, optional
        The cosmology: one of the realizations of astropy (e.g.
        'Planck18', the default, or 'WMAP9').

    Raises
    ------
    ConfigError
        If the cosmology is unknown.
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type(), cosmology=self._cosmology_name)

    def __init__(self, cosmology: str = COSMOLOGY_DEFAULT):
        self._cosmology = _cosmology(cosmology)
        self._cosmology_name = cosmology

    def cosmology(self) -> astropy.cosmology.Cosmology:
        """Return the cosmology (of astropy)."""
        return self._cosmology

    def cosmology_name(self) -> str:
        """Return the name of the cosmology."""
        return self._cosmology_name

    @abc.abstractmethod
    def pdescs(self) -> dict[str, ParamScalarDesc]:
        """Return its parameters, by name."""
        pass

    @abc.abstractmethod
    def kpc_per_arcsec(self, params: ParamValues, redshift: float) -> float:
        """
        Return the length (kpc) of an arcsec at the galaxy.

        Parameters
        ----------
        params : ParamValues
            The values of its parameters (and possibly others), by name.
        redshift : float
            The redshift of the galaxy (see Galaxy).

        Returns
        -------
        float
            The length.
        """
        pass

    @abc.abstractmethod
    def epoch(self, params: ParamValues, redshift: float) -> float:
        """
        Return the redshift of the epoch of the galaxy, at which the
        quantities that depend on the cosmology are evaluated.

        Parameters
        ----------
        params : ParamValues
            The values of its parameters (and possibly others), by name.
        redshift : float
            The redshift of the galaxy (see Galaxy).

        Returns
        -------
        float
            The redshift.
        """
        pass

    def critical_density(self, params: ParamValues, redshift: float) -> float:
        """
        Return the critical density of the universe (Msun/kpc^3) at the
        epoch of the galaxy (see epoch for the arguments).
        """
        return _critical_density(
            self._cosmology_name, self.epoch(params, redshift))

    def omega_m(self, params: ParamValues, redshift: float) -> float:
        """
        Return the matter density parameter at the epoch of the galaxy
        (see epoch for the arguments).
        """
        return _omega_m(self._cosmology_name, self.epoch(params, redshift))

    def ricci_scalar(self, params: ParamValues, redshift: float) -> float:
        """
        Return the Ricci scalar of the background universe (kpc^-2) at the
        epoch of the galaxy (see epoch for the arguments).
        """
        return _ricci_scalar(
            self._cosmology_name, self.epoch(params, redshift))

    def ricci_scalar_today(self) -> float:
        """Return the Ricci scalar of the background universe today."""
        return _ricci_scalar(self._cosmology_name, 0.0)

    def baryon_fraction(self) -> float | None:
        """
        Return the cosmic baryon fraction, Omega_b / Omega_m (the same at
        every epoch; None if the cosmology has no Omega_b).
        """
        return _baryon_fraction(self._cosmology_name)


class DistanceDirect(Distance):
    """
    A distance given directly: its parameter is the angular diameter
    distance (Mpc), and the epoch is the redshift of the galaxy (see
    Galaxy; 0 without one).

    Parameters
    ----------
    cosmology : str, optional
        The cosmology (see Distance).

    Raises
    ------
    ConfigError
        If the cosmology is unknown.
    """

    @staticmethod
    def type() -> str:
        return 'direct'

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(distance=ParamScalarDesc(
            desc="the angular diameter distance of the galaxy (Mpc)",
            minimum=0))

    def kpc_per_arcsec(self, params: ParamValues, redshift: float) -> float:
        return float(params['distance']) * KPC_PER_ARCSEC_MPC

    def epoch(self, params: ParamValues, redshift: float) -> float:
        return redshift


class DistanceRedshift(Distance):
    """
    A distance given by a redshift: the angular diameter distance in the
    cosmology at the redshift of the galaxy (see Galaxy), which is also
    the epoch; or at the given redshift, for the distance only (e.g. one
    corrected for the motion of the galaxy, whose lines show its
    heliocentric redshift). It has no parameters: the galaxy has one
    redshift, and the velocity offset is the systemic velocity of its
    components (vsys).

    Parameters
    ----------
    redshift : float, optional
        The redshift of the distance and the epoch, if not that of the
        galaxy.
    cosmology : str, optional
        The cosmology (see Distance).

    Raises
    ------
    ConfigError
        If the redshift is negative, or the cosmology is unknown.
    """

    @staticmethod
    def type() -> str:
        return 'redshift'

    def dump(self) -> dict[str, Any]:
        info = super().dump()
        if self._redshift is not None:
            info.update(redshift=self._redshift)
        return info

    def __init__(
            self,
            redshift: float | None = None,
            cosmology: str = COSMOLOGY_DEFAULT
    ):
        super().__init__(cosmology)
        if redshift is not None and not redshift >= 0:
            raise ConfigError(
                f"redshift cannot be negative; it is {redshift}")
        self._redshift = None if redshift is None else float(redshift)

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return {}

    def kpc_per_arcsec(self, params: ParamValues, redshift: float) -> float:
        distance = _angular_diameter_distance(
            self.cosmology_name(), self.epoch(params, redshift))
        return distance * KPC_PER_ARCSEC_MPC

    def epoch(self, params: ParamValues, redshift: float) -> float:
        return self._redshift if self._redshift is not None else redshift


distance_parser = parseutils.TypedParser(Distance, [
    DistanceDirect,
    DistanceRedshift])
