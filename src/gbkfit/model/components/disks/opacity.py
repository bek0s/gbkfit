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
    (S the source function), so that the light j of a voxel has the
    optical depth j/S, with the shape of its line. Seen only along a
    Transfer line of sight, by the observations that see the component,
    which must be a thick disk that emits lines.

    With a unit of its emission (see LineEmission), S is the brightness
    (radiation) temperature of the source function, J(T_ex) =
    (h nu / k) / (exp(h nu / k T_ex) - 1), in K, whatever the unit of the
    data: the coefficients are 1/S in 1/K (and j is the light of the
    voxel as brightness temperature, at the frequency of the line at the
    systemic velocity of the component; see _units.OpacityScales).
    Without a unit, S is in the units of the model, those of the data.

    S is J(T_ex), not the excitation temperature T_ex: they differ where
    h nu / k is not small next to T_ex. For HI (h nu / k = 0.068 K) J is
    T_ex (the spin temperature) to 0.03% at 100 K; for CO 3-2 (h nu / k =
    16.6 K), J(10 K) = 3.9 K. To fit T_ex, give it a parameter of the
    user (pdescs) and the coefficient the expression of 1/J, e.g. for CO
    3-2: '(np.exp(16.6 / tex) - 1) / 16.6'. (The background, e.g. the
    CMB, is left out: the light here saturates at S, whereas the light
    above a background of temperature T_bg would saturate at S -
    J(T_bg).) The original GBKFIT had no line opacity: its lines are all
    optically thin, as they are here without it.

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
