"""
Mass components given by a table of their circular velocity (e.g. the
curves of the gas, the disk and the bulge of a galaxy, from its
photometry).
"""

from typing import Any, Self

import astropy.units as u
import numpy as np
import numpy.typing as npt
import scipy.interpolate

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import parseutils, tableutils, unitutils
from gbkfit.utils.parseutils import ConfigError
from .base import MassComponent, MassContext


__all__ = [
    'MassTable'
]


# The relative rounding of the radii of a table, as the distances turn
# them into those of the galaxy and back (see MassTable.vcirc2)
_RADIUS_ROUNDING = 1e-12


class MassTable(MassComponent):
    """
    A mass component given by a table of its circular velocity V in its
    plane at radii (e.g. the curves of the gas, the disk and the bulge of
    SPARC, Vgas, Vdisk and Vbul, of Lelli et al. 2016), times its
    parameter upsilon, the ratio of its mass to that of the table (e.g.
    the mass-to-light ratio of a curve of a stellar disk of upsilon 1):
    v^2 = upsilon V |V|, negative where V is (where it pushes outwards,
    e.g. the gas inside a ring).

    Its radii are angles or lengths. Lengths measured at a distance D0
    (e.g. the distance of the photometry that made the table) follow the
    distance D of the galaxy as its light does: its radii are times D/D0,
    and so is V^2 (the mass of the light is times (D/D0)^2), as SPARC fits
    take them (Li et al. 2018). Angles are lengths at the distance of the
    galaxy, and V^2 is times D/D0 too if D0 is given; without D0, V^2 is as
    given, and lengths are physical radii (e.g. in the Milky Way).

    Between its radii, V |V| is the cubic spline through them, exact at
    them, from V = 0 at the centre (a radius of 0 is added if the table
    has none); beyond its last radius, it has no values (ValueError).

    It gives its circular velocity in its plane only: no enclosed mass,
    density, forces off its plane or potential (NotImplementedError), as
    the gravities and the features that take those need (e.g. f(R)
    gravity, the contraction of a halo, or the jeans equilibrium of a
    disk; for them, give the mass of the photometry instead, e.g. disk
    surfaces 'nodewise').

    In its configuration, the table is the file of one (see
    tableutils.read_table), its format, the columns of its radii (radius:
    {column, unit}) and of its velocities (velocity: {column, unit}; km/s
    by default), and the distance D0 (Mpc), if any; or the radii (a list),
    their unit (radius_unit), the velocities (km/s) and the distance.

    Parameters
    ----------
    name : str
        Its name (see MassComponent).
    radii : Quantity
        The radii of the table: angles or lengths, finite and at least 0,
        increasing.
    velocities : Quantity
        The circular velocity at each radius.
    distance : Quantity, optional
        The distance the table was made at (D0).

    Raises
    ------
    ConfigError
        If the radii are not angles or lengths, finite, at least 0 and
        increasing, the velocities are not velocities, finite and one per
        radius, or the distance is not a positive length.
    """

    @staticmethod
    def type() -> str:
        return 'table'

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        info = dict(info)
        if 'file' in info:
            table = tableutils.read_table(
                info.pop('file'), info.pop('format', None))
            columns = {}
            for key, default in (('radius', None), ('velocity', 'km/s')):
                with parseutils.config_path(key):
                    column = parseutils.parse_options(
                        info.pop(key, None), required={'column'},
                        optional={'unit'})
                    columns[key] = tableutils.read_column(
                        table, column['column'], column.get('unit', default))
            info['radii'], info['velocities'] = \
                columns['radius'], columns['velocity']
        elif 'radii' in info:
            unit = info.pop('radius_unit', None)
            if unit is None:
                raise ConfigError(
                    "the radii need their unit (radius_unit, e.g. kpc)")
            info['radii'] = np.asarray(info['radii'], dtype=float) \
                * unitutils.parse_unit(unit)
            info['velocities'] = np.asarray(
                info.get('velocities'), dtype=float) * u.km / u.s
        if info.get('distance') is not None:
            info['distance'] = info['distance'] * u.Mpc
        return super().load(info)

    def dump(self) -> dict[str, Any]:
        info = super().dump() | dict(
            radii=self._radii.value.tolist(),
            radius_unit=unitutils.format_unit(self._radii.unit),
            velocities=self._velocities.to_value(u.km / u.s).tolist())
        if self._distance is not None:
            info['distance'] = float(self._distance.to_value(u.Mpc))
        return info

    def __init__(
            self,
            name: str,
            radii: u.Quantity,
            velocities: u.Quantity,
            distance: u.Quantity | None = None
    ):
        super().__init__(name)
        if not isinstance(radii, u.Quantity) or radii.ndim != 1 \
                or not (radii.unit.is_equivalent(u.arcsec)
                        or radii.unit.is_equivalent(u.kpc)):
            raise ConfigError(
                f"the radii are a vector of angles (e.g. arcsec) or lengths "
                f"(e.g. kpc), with their unit; they are {radii!r}")
        if not np.all(np.isfinite(radii) & (radii >= 0)) \
                or np.any(np.diff(radii) <= 0):
            raise ConfigError(
                f"the radii must be finite, at least 0 and increasing; they "
                f"are {radii}")
        if not isinstance(velocities, u.Quantity) \
                or not velocities.unit.is_equivalent(u.km / u.s) \
                or velocities.shape != radii.shape \
                or not np.all(np.isfinite(velocities)):
            raise ConfigError(
                f"the velocities must be finite velocities, one per radius; "
                f"they are {velocities!r}")
        if distance is not None and (
                not isinstance(distance, u.Quantity) or distance.size != 1
                or not distance.unit.is_equivalent(u.Mpc)
                or not distance > 0):
            raise ConfigError(
                f"the distance must be a positive length; it is "
                f"{distance!r}")
        self._radii = radii.astype(float)
        self._velocities = velocities.astype(float)
        self._distance = distance
        # The spline of V |V| ((km/s)^2) in the radii of the table, from 0
        x = self._radii.value
        v = self._velocities.to_value(u.km / u.s)
        if x[0] > 0:
            x, v = np.concatenate([[0.0], x]), np.concatenate([[0.0], v])
        self._spline = scipy.interpolate.CubicSpline(x, v * np.abs(v))

    def radii(self) -> u.Quantity:
        """Return the radii of the table."""
        return self._radii

    def velocities(self) -> u.Quantity:
        """Return the circular velocities of the table."""
        return self._velocities

    def distance(self) -> u.Quantity | None:
        """Return the distance the table was made at, if given."""
        return self._distance

    def pdescs(self) -> dict[str, ParamDesc]:
        return dict(upsilon=ParamScalarDesc(
            desc="the ratio of its mass to that of its table (e.g. a "
                 "mass-to-light ratio)"))

    def gives_enclosed_mass(self) -> bool:
        return False

    def gives_forces(self) -> bool:
        return False

    def vcirc2(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        x, scale = self._table_radii(R, context)
        # (radii of the table that the distances turned into others round
        # off: the last one can come out a hair beyond itself)
        last = self._spline.x[-1]
        if np.any(x > last * (1 + _RADIUS_ROUNDING)):
            raise ValueError(
                f"the mass component '{self.name()}' has a table to the "
                f"radius {last:g} {self._radii.unit}; it was asked for radii "
                f"up to {np.max(x):g}")
        return params['upsilon'] * scale * self._spline(np.minimum(x, last))

    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        raise NotImplementedError(
            f"the mass component 'table' ({self.name()}) gives its circular "
            f"velocity in its plane only, not its enclosed mass")

    def _table_radii(
            self, R: np.ndarray, context: MassContext
    ) -> tuple[np.ndarray, float]:
        """
        Return the radii R of the galaxy (kpc) as radii of the table (in
        its unit), and the factor of its V^2 at the distance of the
        galaxy, D/D0 (1 without D0; see the class).

        Raises
        ------
        ValueError
            If the context does not have the length of an arcsec, which
            they need.
        """
        angles = self._radii.unit.is_equivalent(u.arcsec)
        if not angles and self._distance is None:
            return R / self._radii.unit.to(u.kpc), 1.0
        kpc_per_arcsec = context.kpc_per_arcsec
        if kpc_per_arcsec is None:
            raise ValueError(
                f"the mass component '{self.name()}' needs the length of an "
                f"arcsec in its galaxy, which its context does not give")
        arcsec = R / kpc_per_arcsec
        scale = 1.0
        if self._distance is not None:
            # (D/D0, of the lengths of an arcsec at D and D0)
            scale = kpc_per_arcsec / (self._distance * u.arcsec).to_value(
                u.kpc, u.dimensionless_angles())
        if angles:
            return arcsec / self._radii.unit.to(u.arcsec), scale
        # (the lengths of the table, at D0, of those angles)
        lengths = (arcsec * u.arcsec * self._distance).to(
            self._radii.unit, u.dimensionless_angles())
        return lengths.value, scale
