from gbkfit.utils import parseutils
from . import _component, _point
from .base import BrightnessComponent2D


__all__ = [
    'BrightnessPoint2D'
]


class BrightnessPoint2D(BrightnessComponent2D):
    """
    An unresolved component (e.g. a nucleus): its flux at a point (see
    _point.PointPlan). Its parameters: xpos, ypos and flux.
    """

    @staticmethod
    def type():
        return 'point'

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'gmodel component')
        return cls(**parseutils.parse_options_for_callable(
            info, desc, cls.__init__))

    def dump(self):
        return dict(type=self.type()) | _component.dump_name(self)

    def __init__(self, name: str | None = None):
        super().__init__(name)

    def pdescs(self):
        return _point.point_pdescs(spectral=False)

    def plan(self, driver, spectral, dtype, lines):
        return _point.PointPlan(driver, dtype)
