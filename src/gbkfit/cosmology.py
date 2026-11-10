"""
The cosmology of a setup: the background universe of the epochs of its
galaxies, of their distances given by their redshifts (see
model.distance) and of the quantities of their mass and gravity that
depend on it (e.g. the overdensities that define the masses of halos).
It is a published one (a realization of astropy: Planck18 by default) or
a flat Lambda-CDM one of given parameters.

A galaxy evaluated on its own, without a setup (e.g. to preview it),
takes the cosmology by default (DEFAULT).
"""

import abc
from typing import Annotated, Any, Literal, TypeAlias

import astropy.constants
import astropy.cosmology
import numpy as np
from annotated_types import Gt, Interval

from gbkfit.config import (
    Choices, TypedParser, TypedSerializable, Unit, check_options)


__all__ = [
    'DEFAULT',
    'Cosmology',
    'CosmologyFlatLambdaCDM',
    'CosmologyPublished',
    'cosmology_parser'
]


# The published cosmologies: the realizations of astropy
_PUBLISHED = tuple(astropy.cosmology.realizations.available)

# The titles of the published cosmologies (of those astropy may add,
# their names)
_TITLES = dict(
    WMAP1='WMAP 1-year', WMAP3='WMAP 3-year', WMAP5='WMAP 5-year',
    WMAP7='WMAP 7-year', WMAP9='WMAP 9-year', Planck13='Planck 2013',
    Planck15='Planck 2015', Planck18='Planck 2018')

# The option of a published cosmology
_Published: TypeAlias = Annotated[Literal[_PUBLISHED], Choices({
    name: _TITLES.get(name, name) for name in _PUBLISHED})]

# The speed of light (km/s), and the gravitational constant (kpc
# (km/s)^2 / Msun)
_C = astropy.constants.c.to_value('km/s')
_G = astropy.constants.G.to_value('kpc km2 / (s2 solMass)')


class Cosmology(TypedSerializable, abc.ABC):
    """
    A cosmology: the quantities of the background universe at the epochs
    of galaxies (their redshifts), of the cosmology of astropy that it
    is (see astropy_cosmology).
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    @abc.abstractmethod
    def astropy_cosmology(self) -> astropy.cosmology.FLRW:
        """Return the cosmology of astropy that it is."""
        pass

    def h(self) -> float:
        """Return its Hubble constant in units of 100 km/s/Mpc."""
        return float(self.astropy_cosmology().h)

    def angular_diameter_distance(self, redshift: float) -> float:
        """Return the angular diameter distance (Mpc) at a redshift."""
        return float(self.astropy_cosmology().angular_diameter_distance(
            redshift).to_value('Mpc'))

    def critical_density(self, redshift: float) -> float:
        """Return the critical density (Msun/kpc^3) at a redshift."""
        return float(self.astropy_cosmology().critical_density(
            redshift).to_value('solMass / kpc3'))

    def omega_m(self, redshift: float) -> float:
        """
        Return the matter density parameter at a redshift, with the
        massive neutrinos (which astropy keeps apart from the matter), as
        Planck's Omega_m has them.
        """
        cosmology = self.astropy_cosmology()
        return float(cosmology.Om(redshift) + cosmology.Onu(redshift))

    def ricci_scalar(self, redshift: float) -> float:
        """
        Return the Ricci scalar of the background universe (kpc^-2) at a
        redshift: 8 pi G (rho_m + (1 - 3 w) rho_de) / c^2, of its matter
        (with the massive neutrinos, see omega_m) and its dark energy
        (radiation is traceless).
        """
        cosmology = self.astropy_cosmology()
        density = self.critical_density(redshift) * (
            self.omega_m(redshift)
            + (1 - 3 * cosmology.w(redshift)) * cosmology.Ode(redshift))
        return float(8 * np.pi * _G * density / _C ** 2)

    def baryon_fraction(self) -> float | None:
        """
        Return the cosmic baryon fraction, Omega_b / Omega_m (the same at
        every epoch; None if it has no Omega_b), of Omega_m with the
        massive neutrinos (see omega_m).
        """
        # (astropy has 0 or None of none, by its version)
        baryons = self.astropy_cosmology().Ob0
        return float(baryons / self.omega_m(0.0)) if baryons else None


class CosmologyPublished(Cosmology):
    """
    A published cosmology: one of the realizations of astropy, by name
    (e.g. 'Planck18', the default: Planck Collaboration 2020, Table 2;
    or 'WMAP9').

    Parameters
    ----------
    name : str, optional
        Its name.

    Raises
    ------
    ValueError
        If the name is not that of a realization.
    """

    @staticmethod
    def type() -> str:
        return 'published'

    @staticmethod
    def type_title() -> str:
        return 'Published'

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(name=self._name)

    def __init__(self, name: _Published = 'Planck18'):
        check_options(CosmologyPublished.__init__, locals())
        self._name = name
        self._cosmology = getattr(astropy.cosmology.realizations, name)

    def name(self) -> str:
        """Return its name."""
        return self._name

    def astropy_cosmology(self) -> astropy.cosmology.FLRW:
        return self._cosmology


class CosmologyFlatLambdaCDM(Cosmology):
    """
    A flat Lambda-CDM cosmology of given parameters: its Hubble constant
    H0, and its matter density parameter Omega_m (all of its matter) and
    baryon density parameter Omega_b today; no radiation.

    Parameters
    ----------
    hubble_constant : float
        H0 (km/s/Mpc).
    omega_matter : float
        Omega_m, in (0, 1].
    omega_baryon : float, optional
        Omega_b, in (0, Omega_m]; without it, no baryon fraction (which
        the contraction cautun20 takes).

    Raises
    ------
    ValueError
        If the parameters are not valid.
    """

    @staticmethod
    def type() -> str:
        return 'flat_lambda_cdm'

    @staticmethod
    def type_title() -> str:
        return 'Flat Lambda-CDM'

    def dump(self) -> dict[str, Any]:
        info = super().dump() | dict(
            hubble_constant=self._hubble_constant,
            omega_matter=self._omega_matter)
        if self._omega_baryon is not None:
            info['omega_baryon'] = self._omega_baryon
        return info

    def __init__(
            self,
            hubble_constant: Annotated[float, Gt(0), Unit('km / (s Mpc)')],
            omega_matter: Annotated[float, Interval(gt=0, le=1)],
            omega_baryon: Annotated[float, Gt(0)] | None = None
    ):
        check_options(CosmologyFlatLambdaCDM.__init__, locals())
        if omega_baryon is not None and not omega_baryon <= omega_matter:
            raise ValueError(
                f"omega_baryon ({omega_baryon}) cannot exceed omega_matter "
                f"({omega_matter})")
        self._hubble_constant = hubble_constant
        self._omega_matter = omega_matter
        self._omega_baryon = omega_baryon
        baryons = {} if omega_baryon is None else dict(Ob0=omega_baryon)
        self._cosmology = astropy.cosmology.FlatLambdaCDM(
            H0=hubble_constant, Om0=omega_matter, **baryons)

    def astropy_cosmology(self) -> astropy.cosmology.FLRW:
        return self._cosmology


cosmology_parser = TypedParser(Cosmology, [
    CosmologyPublished,
    CosmologyFlatLambdaCDM], family='cosmology')

# The cosmology of galaxies evaluated on their own, without a setup
DEFAULT = CosmologyPublished()
