import abc
from collections.abc import Sequence

import astropy.constants
import astropy.units
import numpy as np
import scipy.special

from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'MassComponent',
    'MassExponentialDisk',
    'MassPseudoIsothermal',
    'MassModel',
    'mass_component_parser',
    'mass_model_parser'
]


# The gravitational constant (kpc (km/s)^2 / Msun)
G = astropy.constants.G.to_value('kpc km2 / (s2 solMass)')

# The length (kpc) of one arcsec at a distance of one Mpc
KPC_PER_ARCSEC_MPC = (1 * astropy.units.arcsec).to_value('rad') * 1e3


class MassComponent(parseutils.TypedSerializable, abc.ABC):
    """
    A component of a mass model (e.g. the stars or the dark matter halo of
    a galaxy): its parameters, and the square of its circular velocity.
    Radii are in kpc, masses in Msun and velocities in km/s.
    """

    def dump(self):
        return dict(type=self.type(), name=self._name)

    def __init__(self, name: str):
        parseutils.check_name(name)
        self._name = name

    def name(self) -> str:
        return self._name

    @abc.abstractmethod
    def pdescs(self) -> dict[str, ParamScalarDesc]:
        pass

    @abc.abstractmethod
    def vcirc2(self, radius: np.ndarray, params: dict) -> np.ndarray:
        """
        The square of the circular velocity ((km/s)^2) at the given radii
        (kpc), for the given parameters (by their names in pdescs).
        """
        pass


class MassExponentialDisk(MassComponent):
    """
    A razor-thin exponential disk (Freeman 1970) of total mass 10^logm
    (Msun) and scale length rd (kpc).
    """

    @staticmethod
    def type():
        return 'exponential_disk'

    def pdescs(self):
        return dict(
            logm=ParamScalarDesc('logm', desc="log10 of the mass (Msun)"),
            rd=ParamScalarDesc('rd', desc="the scale length (kpc)"))

    def vcirc2(self, radius, params):
        # 2 G M / rd y^2 (I0 K0 - I1 K1), with y = r / (2 rd); the scaled
        # Bessel functions keep the products finite at large y, and the
        # limit at y = 0 is 0
        mass = 10 ** params['logm']
        rd = params['rd']
        y = np.asarray(radius, dtype=float) / (2 * rd)
        y_safe = np.where(y > 0, y, 1)
        bessel = (
            scipy.special.i0e(y_safe) * scipy.special.k0e(y_safe)
            - scipy.special.i1e(y_safe) * scipy.special.k1e(y_safe))
        return np.where(y > 0, 2 * G * mass / rd * y_safe ** 2 * bessel, 0)


class MassPseudoIsothermal(MassComponent):
    """
    A pseudo-isothermal sphere of central density 10^logrho0 (Msun/kpc^3)
    and core radius rc (kpc): rho(r) = rho0 / (1 + (r / rc)^2).
    """

    @staticmethod
    def type():
        return 'pseudo_isothermal'

    def pdescs(self):
        return dict(
            logrho0=ParamScalarDesc(
                'logrho0', desc="log10 of the central density (Msun/kpc^3)"),
            rc=ParamScalarDesc('rc', desc="the core radius (kpc)"))

    def vcirc2(self, radius, params):
        # 4 pi G rho0 rc^2 (1 - arctan(x) / x), with x = r / rc; near
        # x = 0, by its series, which does not cancel
        rho0 = 10 ** params['logrho0']
        rc = params['rc']
        x = np.asarray(radius, dtype=float) / rc
        x_safe = np.where(x > 1e-3, x, 1)
        shape = np.where(
            x > 1e-3, 1 - np.arctan(x_safe) / x_safe, x ** 2 / 3 - x ** 4 / 5)
        return 4 * np.pi * G * rho0 * rc ** 2 * shape


mass_component_parser = parseutils.TypedParser(MassComponent, [
    MassExponentialDisk,
    MassPseudoIsothermal])


class MassModel(parseutils.Serializable):
    """
    The mass of a galaxy: its mass components (e.g. stars, gas, dark
    matter), whose circular velocity adds in quadrature. Its parameters
    are the distance of the galaxy (Mpc), which gives the radii of the
    model (arcsec) in kpc, and those of each component, prefixed by
    'mass_' and its name (e.g. mass_halo_rc). The velocity traits 'mass'
    of the components of its model take its circular velocity.
    """

    @classmethod
    def load(cls, info):
        parseutils.load_option_and_update_info(
            mass_component_parser, info, 'components', required=True)
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self):
        return dict(components=mass_component_parser.dump(
            list(self._components)))

    def __init__(self, components: MassComponent | Sequence[MassComponent]):
        components = iterutils.tuplify(components)
        if not components:
            raise ConfigError("a mass model needs at least one component")
        names = [component.name() for component in components]
        if repeated := sorted({n for n in names if names.count(n) > 1}):
            raise ConfigError(
                f"the components of a mass model must have different "
                f"names; repeated: {repeated}")
        params, self._mappings = iterutils.merge_with_prefixes(
            [component.pdescs() for component in components],
            [f'mass_{name}_' for name in names])
        self._components = components
        self._pdescs = dict(distance=ParamScalarDesc(
            'distance', desc="the distance of the galaxy (Mpc)")) | params

    def components(self) -> tuple[MassComponent, ...]:
        return self._components

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return self._pdescs

    def vcirc(self, radius: np.ndarray, params: dict) -> np.ndarray:
        """
        The circular velocity (km/s) at the given radii (arcsec), for the
        given parameters (by their names in pdescs).
        """
        radius_kpc = (
            np.asarray(radius, dtype=float)
            * params['distance'] * KPC_PER_ARCSEC_MPC)
        vcirc2 = sum(
            component.vcirc2(
                radius_kpc, {name: params[mapping[name]] for name in mapping})
            for component, mapping in zip(self._components, self._mappings))
        return np.sqrt(vcirc2)


mass_model_parser = parseutils.BasicParser(MassModel)
