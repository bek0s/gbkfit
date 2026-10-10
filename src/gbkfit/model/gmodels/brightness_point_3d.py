from . import _component, _point
from .base import BrightnessComponent3D


__all__ = [
    'BrightnessPoint3D'
]


class BrightnessPoint3D(BrightnessComponent3D):
    """
    An unresolved component (e.g. a nucleus): its flux at a point (see
    _point.PointPlan). Its parameters: xpos, ypos and flux.
    It is not absorbed by the opacity of the gmodel.
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
