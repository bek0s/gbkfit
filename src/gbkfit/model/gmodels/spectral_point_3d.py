from collections.abc import Sequence

from gbkfit.utils import parseutils
from . import _component, _point
from .base import SpectralComponent3D
from .lines import Line, Lines, line_parser


__all__ = [
    'SpectralPoint3D'
]


class SpectralPoint3D(SpectralComponent3D):
    """
    An unresolved component (e.g. a broad central component): its flux at
    a point, as its emission lines (see _point.PointPlan). Its parameters:
    xpos, ypos, flux, vsys and disp (the dispersion of its lines), and
    the flux ratios of its lines (see Lines).
    It is not absorbed by the opacity of the gmodel.
    """

    @staticmethod
    def type():
        return 'point'

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'gmodel component')
        parseutils.load_option_and_update_info(line_parser, info, 'lines')
        return cls(**parseutils.parse_options_for_callable(
            info, desc, cls.__init__))

    def dump(self):
        return (
            dict(type=self.type())
            | _component.dump_name(self)
            | _component.dump_lines(self._lines))

    def __init__(
            self,
            lines: Sequence[Line] | None = None,
            name: str | None = None
    ):
        super().__init__(name)
        self._lines = Lines(lines)
        pdescs = _point.point_pdescs(spectral=True)
        if repeated := sorted(set(self._lines.pdescs()) & set(pdescs)):
            raise RuntimeError(
                f"the parameters of the lines have the names of other "
                f"parameters: {repeated}; rename the lines")
        self._pdescs = pdescs | self._lines.pdescs()

    def pdescs(self):
        return self._pdescs

    def line_names(self):
        return self._lines.names()

    def plan(self, driver, spectral, dtype, lines):
        selected = self._lines.select(lines)
        values = self._lines.values(spectral)[list(selected)]
        ratios = [
            (row, self._lines.ratio_name(index))
            for row, index in enumerate(selected)
            if self._lines.ratio_name(index) is not None]
        return _point.PointPlan(driver, dtype, values, ratios)
