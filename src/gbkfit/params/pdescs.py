"""
The descriptions of parameters: their size, meaning and valid range.

A description has no name: the name of a parameter is its key in the
dict of parameters it belongs to. Models that combine others give the
parameters of each a prefix (e.g. 'vt' of a trait becomes 'hi_vpt_vt' of
a model), and their dicts share the descriptions.
"""

import abc
import numbers
from collections.abc import Mapping
from typing import Any

import numpy as np

from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'ParamDesc',
    'ParamScalarDesc',
    'ParamVectorDesc',
    'pdesc_parser',
    'load_pdescs_dict',
    'dump_pdescs_dict'
]


class ParamDesc(parseutils.TypedSerializable, abc.ABC):
    """
    The description of a parameter: a scalar, or a vector of elements.

    Parameters
    ----------
    size : int
        The number of its values.
    desc : str, optional
        What it is, for people (e.g. "the core radius (kpc)").
    minimum, maximum : float, optional
        Its valid range: the values for which the models that have it are
        defined (e.g. 0 for a dispersion), not the values that are likely.
        Every element of a vector has the same range.

    Raises
    ------
    ConfigError
        If the size is not a positive integer, or the minimum is above
        the maximum.
    """

    def dump(self) -> dict[str, Any]:
        info: dict[str, Any] = dict(type=self.type())
        if not self.is_scalar():
            info.update(size=self._size)
        if self._desc is not None:
            info.update(desc=self._desc)
        if np.isfinite(self._minimum):
            info.update(minimum=self._minimum)
        if np.isfinite(self._maximum):
            info.update(maximum=self._maximum)
        return info

    def __init__(
            self,
            size: int,
            desc: str | None,
            minimum: float | None,
            maximum: float | None
    ):
        if (isinstance(size, bool) or not isinstance(size, numbers.Integral)
                or size < 1):
            raise ConfigError(
                f"the size of a parameter must be a positive integer; it "
                f"is {size!r}")
        minimum = -np.inf if minimum is None else float(minimum)
        maximum = +np.inf if maximum is None else float(maximum)
        if not minimum <= maximum:
            raise ConfigError(
                f"the minimum of a parameter ({minimum}) cannot be above "
                f"its maximum ({maximum})")
        self._size = int(size)
        self._desc = desc
        self._minimum = minimum
        self._maximum = maximum

    @abc.abstractmethod
    def is_scalar(self) -> bool:
        """Check whether it is a scalar (a vector has elements)."""
        pass

    def size(self) -> int:
        """Return the number of its values (1 for a scalar)."""
        return self._size

    def desc(self) -> str | None:
        """Return what it is, if given."""
        return self._desc

    def minimum(self) -> float:
        """Return the lowest valid value (-inf if none)."""
        return self._minimum

    def maximum(self) -> float:
        """Return the highest valid value (+inf if none)."""
        return self._maximum

    def has_range(self) -> bool:
        """Check whether its valid range is finite on either side."""
        return bool(np.isfinite(self._minimum) or np.isfinite(self._maximum))


class ParamScalarDesc(ParamDesc):
    """
    The description of a scalar parameter.

    Parameters
    ----------
    desc : str, optional
        What it is (see ParamDesc).
    minimum, maximum : float, optional
        Its valid range (see ParamDesc).

    Raises
    ------
    ConfigError
        If the minimum is above the maximum.
    """

    @staticmethod
    def type() -> str:
        return 'scalar'

    def __init__(
            self,
            desc: str | None = None,
            minimum: float | None = None,
            maximum: float | None = None
    ):
        super().__init__(1, desc, minimum, maximum)

    def is_scalar(self) -> bool:
        return True


class ParamVectorDesc(ParamDesc):
    """
    The description of a vector parameter, whose elements are its values
    (e.g. one for each radial node of a disk).

    Parameters
    ----------
    size : int
        The number of its elements.
    desc : str, optional
        What it is (see ParamDesc).
    minimum, maximum : float, optional
        The valid range of each element (see ParamDesc).

    Raises
    ------
    ConfigError
        If the size is not a positive integer, or the minimum is above
        the maximum.
    """

    @staticmethod
    def type() -> str:
        return 'vector'

    def __init__(
            self,
            size: int,
            desc: str | None = None,
            minimum: float | None = None,
            maximum: float | None = None
    ):
        super().__init__(size, desc, minimum, maximum)

    def is_scalar(self) -> bool:
        return False


pdesc_parser = parseutils.TypedParser(ParamDesc, [
    ParamScalarDesc,
    ParamVectorDesc])


def load_pdescs_dict(info: Mapping[str, Any]) -> dict[str, ParamDesc]:
    """
    Load the descriptions of parameters, by name.

    Parameters
    ----------
    info : Mapping
        The configuration of each description, by the name of its
        parameter.

    Returns
    -------
    dict
        The descriptions, by name.

    Raises
    ------
    ConfigError
        If a configuration is invalid.
    """
    return {
        name: parseutils.load_option(
            pdesc_parser.load, info, name, required=True)
        for name in info}


def dump_pdescs_dict(pdescs: Mapping[str, ParamDesc]) -> dict[str, Any]:
    """
    Dump the descriptions of parameters, by name (see load_pdescs_dict).

    Parameters
    ----------
    pdescs : Mapping
        The descriptions, by name.

    Returns
    -------
    dict
        The configuration of each description, by name.
    """
    return {name: pdesc_parser.dump(pdesc) for name, pdesc in pdescs.items()}
