"""
The line opacity of disk components (LineOpacity): the opacity of a
component at its own lines, of coefficients that are traits of its disk.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Self

from gbkfit.utils import iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from ..roles import Opacity, opacity_parser
from .traits import RPTrait, rpt_parser


__all__ = [
    'LineOpacity'
]


class LineOpacity(Opacity):
    """
    The opacity of a component at its own lines (self-absorption, e.g. of
    HI): at each of its lines, its light there times a coefficient, 1/S
    (S the source function, e.g. the spin temperature of HI, in the units
    of the model), so that the light j of a voxel has the optical depth
    j/S, with the shape of its line. Seen only along a Transfer line of
    sight, by the observations that see the component, which must be a
    thick disk that emits lines.

    Parameters
    ----------
    coefficients : RPTrait, Sequence of RPTrait, or Mapping of str to them
        The coefficient of each line: density polar traits, which add up
        (so it can vary in the disk). For an emission of one line without
        a name, its traits (parameters 'op_*', 'op1_*', ...); for named
        lines, a mapping of each line to its traits (parameters
        'op_{line}_*', 'op1_{line}_*', ...). Every line needs one (0 for
        a line that does not absorb; see the disks).

    Raises
    ------
    ConfigError
        If there are no traits, or a line has none.
    """

    @staticmethod
    def type() -> str:
        return 'line'

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        # (the traits of one line, or a mapping of lines to theirs)
        coefficients = info.get('coefficients')
        if coefficients is not None:
            with parseutils.config_path('coefficients'):
                if isinstance(coefficients, Mapping) \
                        and not isinstance(coefficients.get('type'), str):
                    loaded = {}
                    for name, traits in coefficients.items():
                        with parseutils.config_path(name):
                            loaded[name] = rpt_parser.load(traits)
                    info['coefficients'] = loaded
                else:
                    info['coefficients'] = rpt_parser.load(coefficients)
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        coefficients = self._coefficients
        if isinstance(coefficients, Mapping):
            dumped = {
                name: rpt_parser.dump(list(traits))
                for name, traits in coefficients.items()}
        else:
            dumped = rpt_parser.dump(list(coefficients))
        return dict(type=self.type(), coefficients=dumped)

    def __init__(
            self,
            coefficients: (
                RPTrait | Sequence[RPTrait]
                | Mapping[str, RPTrait | Sequence[RPTrait]])
    ):
        if isinstance(coefficients, Mapping):
            coefficients = {
                name: iterutils.tuplify(traits)
                for name, traits in coefficients.items()}
            if empty := [name for name, traits in coefficients.items()
                         if not traits]:
                raise ConfigError(
                    f"the lines {empty} have no coefficient traits")
        else:
            coefficients = iterutils.tuplify(coefficients)
        if not coefficients:
            raise ConfigError("line opacity needs coefficient traits")
        self._coefficients = coefficients

    def coefficients(
            self
    ) -> tuple[RPTrait, ...] | dict[str, tuple[RPTrait, ...]]:
        """
        Return the coefficient traits: those of one line without a name,
        or those of each named line.
        """
        return self._coefficients


opacity_parser.register(LineOpacity)
