from collections.abc import Sequence

import astropy.constants
import astropy.units
import numpy as np

from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import fitsutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Line',
    'Lines',
    'line_parser'
]


# The speed of light (km/s)
_C = astropy.constants.c.to_value('km/s')


class Line(parseutils.BasicSerializable):
    """
    An emission line of a spectral component: its name, and its rest
    wavelength or frequency (see fitsutils.make_rest).
    """

    @classmethod
    def load(cls, info):
        desc = parseutils.make_basic_desc(cls, 'line')
        return cls(**parseutils.parse_options_for_callable(
            info, desc, cls.__init__))

    def dump(self):
        return dict(name=self._name, rest=str(self._rest))

    def __init__(self, name: str, rest: str | astropy.units.Quantity):
        parseutils.check_name(name)
        self._name = name
        self._rest = fitsutils.make_rest(rest)

    def name(self) -> str:
        return self._name

    def rest(self) -> astropy.units.Quantity:
        return self._rest


line_parser = parseutils.BasicParser(Line)


class Lines:
    """
    The emission lines of a spectral component. Without lines, it has one
    line at the velocity of the spectral axis, whatever the line is. With
    lines, each is at its place on the spectral axis, whose velocities
    refer to a rest (see fitsutils.Coords), and the flux of each line
    after the first, relative to the first, is a parameter
    ('{name}_ratio').
    """

    def __init__(self, lines: Sequence[Line] | None):
        if lines is not None:
            lines = tuple(lines)
            if not lines:
                raise RuntimeError("lines needs at least one line")
            names = [line.name() for line in lines]
            if repeated := sorted({n for n in names if names.count(n) > 1}):
                raise RuntimeError(
                    f"the lines must have different names; repeated: "
                    f"{repeated}")
        self._lines = lines
        self._ratios = {} if lines is None else {
            f'{line.name()}_ratio': ParamScalarDesc(
                f'{line.name()}_ratio',
                desc=f"the flux of line {line.name()} relative to line "
                     f"{lines[0].name()}")
            for line in lines[1:]}

    def lines(self) -> tuple[Line, ...] | None:
        return self._lines

    def names(self) -> tuple[str, ...]:
        """The names of the lines (none without lines)."""
        return () if self._lines is None else tuple(
            line.name() for line in self._lines)

    def pdescs(self):
        return self._ratios

    def select(self, names) -> tuple[int, ...]:
        """
        The indices of the lines of the given names (all if None). The
        one line of a component without lines is always selected. Raise
        ConfigError if none of the lines is selected.
        """
        if self._lines is None:
            return (0,)
        if names is None:
            return tuple(range(len(self._lines)))
        selected = tuple(
            i for i, name in enumerate(self.names()) if name in names)
        if not selected:
            raise ConfigError(
                f"a component of the lines {list(self.names())} has none of "
                f"the selected lines {list(names)}; leave it out with the "
                f"components of the observation")
        return selected

    def ratio_name(self, index: int) -> str | None:
        """
        The name of the flux ratio of the line of the given index: none for
        the first line, whose flux is that of the component.
        """
        return None if index == 0 else f'{self.names()[index]}_ratio'

    def values(self, spectral: fitsutils.Grid) -> np.ndarray:
        """
        The offset, scale and flux of each line on a spectral axis (a grid
        of one axis; an array of shape (nlines, 3); see DiskPlan.evaluate),
        such that a line of velocity v is at offset + scale * v on the
        axis, with a flux of 1 (the ratios are parameters).

        With optical velocities (a rest wavelength), a line of rest
        wavelength w at velocity v is at w (1 + v / c), which is at the
        velocity c (k - 1) + k v of the axis, where k is w over the rest of
        the axis. With radio velocities (a rest frequency), a line of rest
        frequency f at velocity v is at f (1 - v / c), which is at the
        velocity c (1 - k) + k v, where k is f over the rest.
        """
        if self._lines is None:
            return np.array([[0.0, 1.0, 1.0]])
        rest = spectral.coords.rest
        if rest is None:
            raise RuntimeError(
                "the lines of a component need the rest wavelength or "
                "frequency that the velocities of the spectral axis refer "
                "to: RESTWAV or RESTFRQ in the header of the data, or the "
                "rest (scube, lslit) or spec_rest option of the observable")
        optical = rest.unit == astropy.units.m
        values = []
        for line in self._lines:
            ratio = float(line.rest().to_value(
                rest.unit, astropy.units.spectral()) / rest.value)
            offset = _C * (ratio - 1) if optical else _C * (1 - ratio)
            values.append((offset, ratio, 1.0))
        return np.array(values)
