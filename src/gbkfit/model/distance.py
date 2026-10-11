"""
The distance of a galaxy: the length of an arcsec at the galaxy, and the
epoch of the quantities of the galaxy that depend on the cosmology (e.g.
the overdensities that define the mass of a halo).

Inside, the distance is always the angular diameter distance. It is
given directly (as a parameter, in Mpc), or by a redshift (a parameter)
and a cosmology, which give it. A redshift never follows from a given
distance: for nearby galaxies, peculiar velocities make it wrong.
"""

import abc
import functools
from typing import Any

import astropy.cosmology
import astropy.units

from gbkfit.params import ParamValues
from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError


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
    def kpc_per_arcsec(self, params: ParamValues) -> float:
        """
        Return the length (kpc) of an arcsec at the galaxy.

        Parameters
        ----------
        params : ParamValues
            The values of its parameters (and possibly others), by name.

        Returns
        -------
        float
            The length.
        """
        pass

    @abc.abstractmethod
    def epoch(self, params: ParamValues) -> float:
        """
        Return the redshift of the epoch of the galaxy, at which the
        quantities that depend on the cosmology are evaluated.

        Parameters
        ----------
        params : ParamValues
            The values of its parameters (and possibly others), by name.

        Returns
        -------
        float
            The redshift.
        """
        pass

    def critical_density(self, params: ParamValues) -> float:
        """
        Return the critical density of the universe (Msun/kpc^3) at the
        epoch of the galaxy (see epoch for the arguments).
        """
        return _critical_density(self._cosmology_name, self.epoch(params))

    def omega_m(self, params: ParamValues) -> float:
        """
        Return the matter density parameter at the epoch of the galaxy
        (see epoch for the arguments).
        """
        return _omega_m(self._cosmology_name, self.epoch(params))


class DistanceDirect(Distance):
    """
    A distance given directly: its parameter is the angular diameter
    distance (Mpc), and the epoch is a fixed redshift.

    Parameters
    ----------
    epoch : float, optional
        The redshift of the epoch of the galaxy (by default 0), at which
        the quantities that depend on the cosmology are evaluated. It
        does not set the distance.
    cosmology : str, optional
        The cosmology (see Distance).

    Raises
    ------
    ConfigError
        If the epoch is negative, or the cosmology is unknown.
    """

    @staticmethod
    def type() -> str:
        return 'direct'

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(epoch=self._epoch)

    def __init__(self, epoch: float = 0, cosmology: str = COSMOLOGY_DEFAULT):
        super().__init__(cosmology)
        if not epoch >= 0:
            raise ConfigError(f"epoch cannot be negative; it is {epoch}")
        self._epoch = float(epoch)

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(distance=ParamScalarDesc(
            desc="the angular diameter distance of the galaxy (Mpc)",
            minimum=0))

    def kpc_per_arcsec(self, params: ParamValues) -> float:
        return float(params['distance']) * KPC_PER_ARCSEC_MPC

    def epoch(self, params: ParamValues) -> float:
        return self._epoch


class DistanceRedshift(Distance):
    """
    A distance given by a redshift: its parameter is the redshift of the
    galaxy, which gives the angular diameter distance in the cosmology,
    and is the epoch.

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
        return 'redshift'

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(redshift=ParamScalarDesc(
            desc="the redshift of the galaxy", minimum=0))

    def kpc_per_arcsec(self, params: ParamValues) -> float:
        distance = _angular_diameter_distance(
            self.cosmology_name(), float(params['redshift']))
        return distance * KPC_PER_ARCSEC_MPC

    def epoch(self, params: ParamValues) -> float:
        return float(params['redshift'])


distance_parser = parseutils.TypedParser(Distance, [
    DistanceDirect,
    DistanceRedshift])
