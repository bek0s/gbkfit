"""
Mass components given by a function of the user.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import funcutils, parseutils
from gbkfit.utils.parseutils import ConfigError
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
    the jeans equilibrium; in its plane, -vcirc2 / R and 0). Without
    forces, it gives them in its plane only. It gives nothing else: its
    density and its potential raise NotImplementedError.

    Parameters
    ----------
    name : str
        Its name (see MassComponent).
    file : str
        The Python file that defines the function (relative to the
        configuration).
    func : str
        The name of the function of the squared circular velocity.
    params : Sequence of str, optional
        The names of its parameters.
    forces : str, optional
        The name of the function of the forces.

    Raises
    ------
    ConfigError
        If the file does not define the functions, or a parameter name is
        not valid or repeats.
    """

    @staticmethod
    def type() -> str:
        return 'user'

    def gives_forces(self) -> bool:
        return self._forces is not None

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(
            file=self._file, func=self._func_name, params=list(self._params),
            forces=self._forces_name)

    def __init__(
            self,
            name: str,
            file: str,
            func: str,
            params: Sequence[str] = (),
            forces: str | None = None
    ):
        super().__init__(name)
        for param in params:
            parseutils.check_name(param)
        names = list(params)
        if repeated := sorted({p for p in names if names.count(p) > 1}):
            raise ConfigError(f"the parameters repeat: {repeated}")
        path = parseutils.resolve_path(file)
        try:
            self._func = funcutils.load_function_from_file(path, func)
            self._forces = None if forces is None \
                else funcutils.load_function_from_file(path, forces)
        except RuntimeError as e:
            raise ConfigError(str(e)) from e
        self._file = file
        self._func_name = func
        self._forces_name = forces
        self._params = tuple(params)

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
                f"the function '{self._func_name}' of the mass component "
                f"'{self.name()}' returned an array of shape {vcirc2.shape} "
                f"for radii of shape {R.shape}")
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
                f"the function '{self._forces_name}' of the mass component "
                f"'{self.name()}' must return two arrays of the shape of the "
                f"points, {R.shape}")
        return forces[0], forces[1]
