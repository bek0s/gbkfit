"""
Axisymmetric mass disks: a surface density (see surfaces) and a
vertical structure (see vertical), each chosen on its own.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from .base import MassComponent, MassContext
from .surfaces import DiskSurface, disk_surface_parser
from .vertical import DiskVertical, disk_vertical_parser


__all__ = [
    'MassDisk'
]


class MassDisk(MassComponent):
    """
    An axisymmetric disk, of a surface density (see surfaces: e.g. an
    exponential, or that of a component of the galaxy) and a vertical
    structure (see vertical: a law, e.g. sech2, or that of a component),
    each chosen on its own. A disk follows at most one component (by its
    surface density, its vertical structure, or both); following it by
    both, it is the density of the component itself (each of its terms of
    the pdf of its height).

    Parameters
    ----------
    name : str
        Its name (see MassComponent).
    surface : DiskSurface
        Its surface density.
    vertical : DiskVertical
        Its vertical structure.

    Raises
    ------
    ConfigError
        If the name is not valid, or the surface density and the vertical
        structure follow different components.
    """

    @staticmethod
    def type() -> str:
        return 'disk'

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'MassDisk':
        parseutils.load_option_and_update_info(
            disk_surface_parser, info, 'surface', required=True)
        parseutils.load_option_and_update_info(
            disk_vertical_parser, info, 'vertical', required=True)
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(
            surface=disk_surface_parser.dump(self._surface),
            vertical=disk_vertical_parser.dump(self._vertical))

    def __init__(
            self, name: str, surface: DiskSurface, vertical: DiskVertical):
        super().__init__(name)
        followed = {
            surface.followed_component(), vertical.followed_component()}
        followed.discard(None)
        if len(followed) > 1:
            raise ConfigError(
                f"a mass disk follows at most one component; its surface "
                f"density follows {surface.followed_component()!r} and its "
                f"vertical structure {vertical.followed_component()!r}")
        self._surface = surface
        self._vertical = vertical
        self._followed = tuple(followed)

    def surface(self) -> DiskSurface:
        """Return its surface density."""
        return self._surface

    def vertical(self) -> DiskVertical:
        """Return its vertical structure."""
        return self._vertical

    def followed_components(self) -> tuple[str, ...]:
        return self._followed

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._surface.pdescs() | self._vertical.pdescs()

    def vcirc2(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self._surface.vcirc2(R, self._vertical, params, context)

    def density(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self._surface.surface_density(R, params, context) \
            * self._vertical.height_pdf(R, z, params, context)
