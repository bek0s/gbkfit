"""
Mass components given by a function of the user.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import funcutils, parseutils
from gbkfit.utils.parseutils import ConfigError, check_options
from gbkfit.utils.typeutils import FilePath
from .base import MassComponent, MassContext


__all__ = [
    'MassUser'
]


class MassUser(MassComponent):
    """
    A mass component given by Python functions: func(R, params), with R
    the radii (kpc, an array) and params the values of its parameters (a
    dict, by name), which returns the squared circular velocity
    ((km/s)^2, signed, an array of the shape of R); and optionally
    forces(R, z, params), with R and z the points (kpc, two arrays of one
    shape), which returns its forces per unit mass there ((km/s)^2/kpc,
    -dPhi/dR and -dPhi/dz, two arrays of their shape), anywhere (e.g. for
    the jeans equilibrium; in its plane, -vcirc2 / R and 0); and
    optionally mass(r, params), with r the radii of spheres about its
    centre (kpc, an array), which returns its mass within them (Msun, an
    array of the shape of r; e.g. for the enclosed masses of the
    functions of the galaxy). Without forces, it gives them in its plane
    only; without mass, no enclosed mass. It gives nothing else: its
    density and its potential raise NotImplementedError.

    Configurations give the functions by a Python file and their names
    (see from_file); one made from functions in Python has no
    configuration (see dump).

    Parameters
    ----------
    name : str
        Its name (see MassComponent).
    func : Callable
        The function of the squared circular velocity.
    params : Sequence of str, optional
        The names of its parameters.
    forces : Callable, optional
        The function of the forces.
    mass : Callable, optional
        The function of the enclosed mass.

    Raises
    ------
    ConfigError
        If a parameter name is not valid or repeats.
    """

    config_constructors = ('from_file',)

    @staticmethod
    def type() -> str:
        return 'user'

    def gives_forces(self) -> bool:
        return self._forces is not None

    def gives_enclosed_mass(self) -> bool:
        return self._mass is not None

    def dump(self) -> dict[str, Any]:
        if self._source is None:
            raise TypeError(
                f"the mass component '{self.name()}' was made from Python "
                f"functions, which have no configuration; make it from a "
                f"file (see MassUser.from_file)")
        file, func, forces, mass = self._source
        return super().dump() | dict(
            file=file, func=func, params=list(self._params), forces=forces,
            mass=mass)

    @classmethod
    def from_file(
            cls,
            name: str,
            file: FilePath,
            func: str,
            params: Sequence[str] = (),
            forces: str | None = None,
            mass: str | None = None
    ) -> 'MassUser':
        """
        Make a mass component of the functions of a Python file.

        Parameters
        ----------
        name : str
            Its name (see MassComponent).
        file : str
            The Python file that defines the functions.
        func : str
            The name of the function of the squared circular velocity.
        params : Sequence of str, optional
            The names of its parameters.
        forces : str, optional
            The name of the function of the forces.
        mass : str, optional
            The name of the function of the enclosed mass.

        Returns
        -------
        MassUser
            The mass component.

        Raises
        ------
        ConfigError
            If the file does not define the functions, or a parameter name
            is not valid or repeats.
        """
        check_options(MassUser.from_file, locals())
        try:
            functions = [
                None if x is None
                else funcutils.load_function_from_file(file, x)
                for x in (func, forces, mass)]
        except RuntimeError as e:
            raise ConfigError(str(e)) from e
        component = cls(name, functions[0], params, *functions[1:])
        component._source = (file, func, forces, mass)
        return component

    def __init__(
            self,
            name: str,
            func: Callable,
            params: Sequence[str] = (),
            forces: Callable | None = None,
            mass: Callable | None = None
    ):
        check_options(MassUser.__init__, locals())
        super().__init__(name)
        for param in params:
            parseutils.check_name(param)
        names = list(params)
        if repeated := sorted({p for p in names if names.count(p) > 1}):
            raise ConfigError(f"the parameters repeat: {repeated}")
        self._func = func
        self._forces = forces
        self._mass = mass
        self._params = tuple(params)
        # (the file and the names of the functions, of one made from a
        # file: its configuration; see from_file and dump)
        self._source: tuple[str, str, str | None, str | None] | None = None

    def pdescs(self) -> dict[str, ParamDesc]:
        return {name: ParamScalarDesc() for name in self._params}

    def vcirc2(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        vcirc2 = np.asarray(self._func(R, dict(params)), dtype=float)
        if vcirc2.shape != R.shape:
            raise ValueError(
                f"the function '{self._func.__name__}' of the mass "
                f"component '{self.name()}' returned an array of shape "
                f"{vcirc2.shape} for radii of shape {R.shape}")
        return vcirc2

    def forces(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._forces is None:
            return super().forces(R, z, params, context)
        R, z = np.broadcast_arrays(
            np.asarray(R, dtype=float), np.asarray(z, dtype=float))
        forces = [
            np.asarray(force, dtype=float)
            for force in self._forces(R, z, dict(params))]
        if len(forces) != 2 or any(f.shape != R.shape for f in forces):
            raise ValueError(
                f"the function '{self._forces.__name__}' of the mass "
                f"component '{self.name()}' must return two arrays of the "
                f"shape of the points, {R.shape}")
        return forces[0], forces[1]

    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        if self._mass is None:
            raise NotImplementedError(
                f"the mass component '{self.name()}' has no function of its "
                f"enclosed mass (see MassUser)")
        r = np.asarray(r, dtype=float)
        mass = np.asarray(self._mass(r, dict(params)), dtype=float)
        if mass.shape != r.shape:
            raise ValueError(
                f"the function '{self._mass.__name__}' of the mass "
                f"component '{self.name()}' returned an array of shape "
                f"{mass.shape} for radii of shape {r.shape}")
        return mass
