"""
The deflections of the masses of galaxies laid out on the sky (see
model.mass.SkyMass), in the units of the lenses, kept layer by layer
between evaluations (see MassDeflections), and the bending of the light
by the mass of a galaxy in an evaluation (see MassBending).
"""

import dataclasses
import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from gbkfit.driver import Driver
from gbkfit.model.mass import SkyLayer, SkyMass
from ._layers import unit_deflection
from .pieces import SkyPoints


__all__ = [
    'MassBending',
    'MassDeflections'
]


# The arcsec of a radian
_ARCSEC = math.degrees(1) * 3600


class _KeptLayer(NamedTuple):
    """
    The deflection of the unit shape of a layer at points (see
    MassDeflections): the layer (whose unit shape it is), and the
    deflection (radians) along x and y.
    """
    layer: SkyLayer
    unit: tuple[np.ndarray, np.ndarray]


class MassDeflections:
    """
    The deflection of the mass of a galaxy on the sky at points, kept
    layer by layer between evaluations (e.g. those of the rays of an
    observation on the plane of its lens, see TracePlan): the deflection
    of the unit shape of each layer (see unit_deflection) is computed
    again only when its unit shape (see SkyLayer.same_shape) or the
    points change; a change of its scale alone (e.g. a mass-to-light
    ratio) only scales it, and a layer of a part whose parameters are
    fixed costs nothing.

    The layers are those of the parts of the mass by name, in their order
    (see SkyMass.parts): a part whose layers change in number is computed
    again.

    Parameters
    ----------
    driver : Driver
        The driver whose native kernels compute the deflections (e.g. that
        of the observation whose rays they are; see unit_deflection).
    """

    def __init__(self, driver: Driver):
        self._driver = driver
        # The points (kpc) of the deflections kept, and the deflection of
        # each layer of each part, by the name of the part
        self._points: tuple[np.ndarray, np.ndarray] | None = None
        self._kept: dict[str, list[_KeptLayer]] = {}

    def deflection(
            self, mass: SkyMass, points: SkyPoints
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the deflection of the mass at points of its plane, as that
        of the pieces of lenses at D_ls / D_s = 1 (alpha-hat; see
        Normalisation): at the points of the mass (kpc) of the length of
        an arcsec in the galaxy, that of its distance (see
        SkyMass.kpc_per_arcsec), of the unit deflections of its layers that
        are kept, and those of the others computed and kept.

        Parameters
        ----------
        mass : SkyMass
            The mass.
        points : SkyPoints
            The points (arcsec, in the frame of the parameters of the
            galaxies, as the layers).

        Returns
        -------
        tuple of np.ndarray
            The deflection (arcsec) along x and y, of the shape of the
            points.
        """
        x = np.asarray(points.x, dtype=float) * mass.kpc_per_arcsec
        y = np.asarray(points.y, dtype=float) * mass.kpc_per_arcsec
        if self._points is None or not (
                np.array_equal(self._points[0], x)
                and np.array_equal(self._points[1], y)):
            self._points = (x, y)
            self._kept = {}
        alpha_x, alpha_y = np.zeros(np.shape(x)), np.zeros(np.shape(y))
        kept = {}
        for name, layers in mass.parts.items():
            kept[name] = self._part(self._kept.get(name, ()), layers)
            for item in kept[name]:
                alpha_x += item.layer.scale * item.unit[0]
                alpha_y += item.layer.scale * item.unit[1]
        self._kept = kept
        return _ARCSEC * alpha_x, _ARCSEC * alpha_y

    def _part(
            self,
            kept: Sequence[_KeptLayer],
            layers: Sequence[SkyLayer]
    ) -> list[_KeptLayer]:
        """
        Return the unit deflections of the layers of a part at the kept
        points: those kept of the same unit shapes, at the scale of the new
        layers, and the others computed.
        """
        x, y = self._points
        if len(kept) != len(layers):
            kept = [None] * len(layers)
        return [
            _KeptLayer(layer, item.unit)
            if item is not None and item.layer.same_shape(layer)
            else _KeptLayer(
                layer, unit_deflection(layer, x, y, self._driver))
            for layer, item in zip(layers, kept)]


@dataclasses.dataclass(frozen=True)
class MassBending:
    """
    The bending of the light by the mass of a galaxy in an evaluation (see
    pieces.GalaxyBending): its mass on the sky, deflected where the
    deflections of its layers are kept (e.g. a new MassDeflections, of a
    driver, to keep none).

    Attributes
    ----------
    mass : SkyMass
        The mass.
    kept : MassDeflections
        Where the deflections of its layers are kept.
    """
    mass: SkyMass
    kept: MassDeflections

    def deflection(self, points: SkyPoints) -> tuple[np.ndarray, np.ndarray]:
        """
        Return its deflection at points of the plane of the galaxy (see
        MassDeflections.deflection): along x and y (arcsec), of the shape
        of the points.
        """
        return self.kept.deflection(self.mass, points)
