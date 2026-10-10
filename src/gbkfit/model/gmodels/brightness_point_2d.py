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

    def dump(self):
        return dict(type=self.type()) | _component.dump_name(self)

    def __init__(self, name: str | None = None):
        super().__init__(name)

    def pdescs(self):
        return _point.point_pdescs(spectral=False)

    def plan(self, driver, spectral, dtype, lines):
        return _point.PointPlan(driver, dtype)
