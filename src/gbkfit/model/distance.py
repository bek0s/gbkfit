"""
The distance of a galaxy: the length of an arcsec at the galaxy.

Inside, the distance is always the angular diameter distance. It is
given as a parameter (in Mpc), or by the redshift of the galaxy (see
Galaxy) in the cosmology of its setup (see gbkfit.cosmology). A redshift
never follows from a given distance: for nearby galaxies, peculiar
velocities make it wrong. The epoch of the quantities of the galaxy that
depend on the cosmology (e.g. the overdensities that define the masses
of halos) is always the redshift of the galaxy.
"""

import abc
from typing import Any

import astropy.units

from gbkfit.config import TypedParser, TypedSerializable
from gbkfit.cosmology import Cosmology
from gbkfit.params import ParamValues
from gbkfit.params.pdescs import ParamScalarDesc


__all__ = [
    'KPC_PER_ARCSEC_MPC',
    'Distance',
    'DistanceCosmological',
    'DistanceParameter',
    'distance_parser'
]


# The length (kpc) of one arcsec at an angular diameter distance of 1 Mpc
KPC_PER_ARCSEC_MPC = (1 * astropy.units.arcsec).to_value('rad') * 1e3


class Distance(TypedSerializable, abc.ABC):
    """
    The distance of a galaxy: its parameters, and the length of an
    arcsec at the galaxy.
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    @abc.abstractmethod
    def pdescs(self) -> dict[str, ParamScalarDesc]:
        """Return its parameters, by name."""
        pass

    @abc.abstractmethod
    def kpc_per_arcsec(
            self,
            params: ParamValues,
            redshift: float,
            cosmology: Cosmology
    ) -> float:
        """
        Return the length (kpc) of an arcsec at the galaxy, of the values
        of its parameters (and possibly others, by name), the redshift of
        the galaxy (see Galaxy) and the cosmology.
        """
        pass

    def needs_redshift(self) -> bool:
        """
        Check whether it needs the galaxy to have a redshift above 0 (see
        Galaxy); here, it does not.
        """
        return False


class DistanceParameter(Distance):
    """
    A distance that is a parameter: the angular diameter distance of the
    galaxy (Mpc).
    """

    @staticmethod
    def type() -> str:
        return 'parameter'

    @staticmethod
    def type_title() -> str:
        return 'A parameter'

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(distance=ParamScalarDesc(
            title='Angular diameter distance',
            desc="the angular diameter distance of the galaxy",
            min=0, unit='Mpc', default=10))

    def kpc_per_arcsec(
            self,
            params: ParamValues,
            redshift: float,
            cosmology: Cosmology
    ) -> float:
        return float(params['distance']) * KPC_PER_ARCSEC_MPC


class DistanceCosmological(Distance):
    """
    A distance given by the redshift of the galaxy (see Galaxy, which
    must have one above 0): the angular diameter distance at it, in the
    cosmology. It has no parameters: the galaxy has one redshift, and the
    velocity offset is the systemic velocity of its components (vsys).
    """

    @staticmethod
    def type() -> str:
        return 'cosmological'

    @staticmethod
    def type_title() -> str:
        return 'From the redshift'

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return {}

    def kpc_per_arcsec(
            self,
            params: ParamValues,
            redshift: float,
            cosmology: Cosmology
    ) -> float:
        return cosmology.angular_diameter_distance(redshift) \
            * KPC_PER_ARCSEC_MPC

    def needs_redshift(self) -> bool:
        return True


distance_parser = TypedParser(Distance, [
    DistanceParameter,
    DistanceCosmological], family='distance')
