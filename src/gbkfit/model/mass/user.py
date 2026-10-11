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
    A mass component given by a Python function, func(R, params): R the
    radii (kpc, an array) and params the values of its parameters (a
    dict, by name), which returns the squared circular velocity
    ((km/s)^2, signed, an array of the shape of R). It gives nothing else:
    its density, its forces off its plane and its potential raise
    NotImplementedError.

    Parameters
    ----------
    name : str
        Its name (see MassComponent).
    file : str
        The Python file that defines the function (relative to the
        configuration).
    func : str
        The name of the function.
    params : Sequence of str, optional
        The names of its parameters.

    Raises
    ------
    ConfigError
        If the file does not define the function, or a parameter name is
        not valid or repeats.
    """

    @staticmethod
    def type() -> str:
        return 'user'

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(
            file=self._file, func=self._func_name, params=list(self._params))

    def __init__(
            self,
            name: str,
            file: str,
            func: str,
            params: Sequence[str] = ()
    ):
        super().__init__(name)
        for param in params:
            parseutils.check_name(param)
        names = list(params)
        if repeated := sorted({p for p in names if names.count(p) > 1}):
            raise ConfigError(f"the parameters repeat: {repeated}")
        try:
            self._func = funcutils.load_function_from_file(
                parseutils.resolve_path(file), func)
        except RuntimeError as e:
            raise ConfigError(str(e)) from e
        self._file = file
        self._func_name = func
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
