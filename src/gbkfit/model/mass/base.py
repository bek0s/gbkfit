"""
The contract of the components of mass models (MassComponent), and the
spherical components (MassSpherical).
"""

import abc
import dataclasses
from typing import Any

import numpy as np
import numpy.typing as npt

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import parseutils
from . import _numerics
from .units import G


__all__ = [
    'MassComponent',
    'MassContext',
    'MassSpherical'
]


@dataclasses.dataclass(frozen=True)
class MassContext:
    """
    What mass components need besides their parameters: the critical
    density of the universe (Msun/kpc^3) and its matter density
    parameter, at the epoch of the galaxy (for the overdensities that
    define the masses of halos).
    """
    critical_density: float
    omega_m: float


class MassComponent(parseutils.TypedSerializable, abc.ABC):
    """
    A component of a mass model (e.g. the stars, the gas or the dark
    matter halo of a galaxy), axisymmetric about the z axis: R is the
    cylindrical radius and z the height above its plane (see units for
    the units).

    A component computes what it can exactly, or numerically to a known
    precision. What it cannot compute (yet) raises NotImplementedError,
    never an approximation (e.g. a disk treated as a sphere).

    Parameters
    ----------
    name : str
        Its name, which prefixes its parameters in the mass model.

    Raises
    ------
    ConfigError
        If the name is not valid (see parseutils.check_name).
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type(), name=self._name)

    def __init__(self, name: str):
        parseutils.check_name(name)
        self._name = name

    def name(self) -> str:
        """Return its name."""
        return self._name

    @abc.abstractmethod
    def pdescs(self) -> dict[str, ParamDesc]:
        """Return its parameters, by name."""
        pass

    @abc.abstractmethod
    def vcirc2(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return the square of its circular velocity in its plane, R dPhi/dR
        at z = 0: signed (negative where it pushes outwards, e.g. inside
        a ring).

        Parameters
        ----------
        R : array_like
            The radii (kpc).
        params : dict
            The values of its parameters, by their names in pdescs.
        context : MassContext
            The epoch of the galaxy.

        Returns
        -------
        np.ndarray
            The squared circular velocity ((km/s)^2) at each radius.
        """
        pass

    def density(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its density (Msun/kpc^3) at the points (R, z), broadcast
        together (see vcirc2 for the other arguments).

        Raises
        ------
        NotImplementedError
            If the component does not give its density.
        """
        raise NotImplementedError(
            f"the mass component '{self.type()}' does not give its density")

    def forces(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return its forces per unit mass ((km/s)^2/kpc), -dPhi/dR and
        -dPhi/dz, at the points (R, z), broadcast together (see vcirc2
        for the other arguments). By default, in its plane only, where
        they are -vcirc2 / R and 0.

        Raises
        ------
        NotImplementedError
            For points off its plane, if the component does not give its
            forces there.
        """
        R, z = np.broadcast_arrays(
            np.asarray(R, dtype=float), np.asarray(z, dtype=float))
        if np.any(z != 0):
            raise NotImplementedError(
                f"the mass component '{self.type()}' gives its forces in "
                f"its plane (z = 0) only")
        force_R = -np.divide(
            self.vcirc2(R, params, context), R,
            out=np.zeros_like(R), where=R > 0)
        return force_R, np.zeros_like(R)

    def potential(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its gravitational potential ((km/s)^2), 0 at infinity, at
        the points (R, z), broadcast together (see vcirc2 for the other
        arguments).

        Raises
        ------
        NotImplementedError
            If the component does not give its potential.
        """
        raise NotImplementedError(
            f"the mass component '{self.type()}' does not give its "
            f"potential")


class MassSpherical(MassComponent, abc.ABC):
    """
    A spherical mass component, given by its density and its enclosed
    mass, as functions of the spherical radius r: its circular velocity
    and its forces follow exactly, and its potential numerically (unless
    a component gives it exactly).
    """

    @abc.abstractmethod
    def density_spherical(
            self,
            r: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """Return its density (Msun/kpc^3) at the spherical radii r."""
        pass

    @abc.abstractmethod
    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its mass (Msun) within the spherical radii r (kpc; see
        vcirc2 for the other arguments).
        """
        pass

    def vcirc2(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        # G M(<r) / r, 0 at the centre
        R = np.asarray(R, dtype=float)
        return G * np.divide(
            self.enclosed_mass(R, params, context), R,
            out=np.zeros_like(R), where=R > 0)

    def density(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self.density_spherical(np.hypot(R, z), params, context)

    def forces(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> tuple[np.ndarray, np.ndarray]:
        # -G M(<r) / r^3 (R, z), 0 at the centre
        R, z = np.broadcast_arrays(
            np.asarray(R, dtype=float), np.asarray(z, dtype=float))
        r = np.hypot(R, z)
        scale = -G * np.divide(
            self.enclosed_mass(r, params, context), r ** 3,
            out=np.zeros_like(r), where=r > 0)
        return scale * R, scale * z

    def potential(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        # -G (M(<r) / r + 4 pi int_r^inf rho r' dr'), by quadrature
        r = np.hypot(R, z)
        return _numerics.spherical_potential(
            lambda x: self.density_spherical(x, params, context),
            lambda x: self.enclosed_mass(x, params, context), r)
