"""
Mass components given by a table of their circular velocity (e.g. the
curves of the gas, the disk and the bulge of a galaxy, from its
photometry).
"""

from collections.abc import Sequence
from typing import Annotated, Any, Self

import astropy.units as u
import numpy as np
import numpy.typing as npt
import scipy.interpolate
from annotated_types import Gt

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import tableutils, unitutils
from gbkfit.utils.parseutils import ConfigError, check_options
from gbkfit.utils.typeutils import Column, FilePath, Unit
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

    Configurations give its table, or the file of one (see from_file).

    Parameters
    ----------
    name : str
        Its name (see MassComponent).
    radii : Quantity
        The radii of the table: angles or lengths, finite and at least 0,
        increasing.
    velocities : Sequence of float
        The circular velocity at each radius (km/s).
    distance : float, optional
        The distance the table was made at, D0 (Mpc).

    Raises
    ------
    ConfigError
        If the radii are not angles or lengths, finite, at least 0 and
        increasing, or the velocities are not finite and one per radius.
    """

    config_constructors = ('__init__', 'from_file')

    @staticmethod
    def type() -> str:
        return 'table'

    @classmethod
    def from_file(
            cls,
            name: str,
            file: FilePath,
            radius: Column,
            velocity: Column,
            format: str | None = None,
            distance: Annotated[float, Gt(0), Unit('Mpc')] | None = None
    ) -> Self:
        """
        Read a mass table from a table file (see tableutils.read_table).

        Parameters
        ----------
        name : str
            Its name (see MassComponent).
        file : str
            The file.
        radius, velocity : Column
            The columns of the radii and of the velocities (by default
            in km/s, if the file gives no unit).
        format : str, optional
            The format of the file (see astropy's Table.read); by
            default, the one astropy tells.
        distance : float, optional
            The distance the table was made at, D0 (Mpc).

        Returns
        -------
        MassTable
            The mass component.
        """
        check_options(MassTable.from_file, locals())
        table = tableutils.read_table(file, format)
        radii = tableutils.read_column(table, radius.column, radius.unit)
        velocities = tableutils.read_column(
            table, velocity.column, velocity.unit or 'km/s')
        return cls(
            name, radii, velocities.to_value(u.km / u.s).tolist(), distance)

    def dump(self) -> dict[str, Any]:
        info = super().dump() | dict(
            radii=dict(
                value=self._radii.value.tolist(),
                unit=unitutils.format_unit(self._radii.unit)),
            velocities=self._velocities.to_value(u.km / u.s).tolist())
        if self._distance is not None:
            info['distance'] = self._distance
        return info

    def __init__(
            self,
            name: str,
            radii: u.Quantity,
            velocities: Annotated[Sequence[float] | np.ndarray, Unit('km/s')],
            distance: Annotated[float, Gt(0), Unit('Mpc')] | None = None
    ):
        check_options(MassTable.__init__, locals())
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
        velocities = np.asarray(velocities, dtype=float)
        if velocities.shape != radii.shape \
                or not np.all(np.isfinite(velocities)):
            raise ConfigError(
                f"the velocities must be finite, one per radius; they are "
                f"{velocities}")
        self._radii = radii.astype(float)
        self._velocities = velocities * (u.km / u.s)
        self._distance = None if distance is None else float(distance)
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

    def distance(self) -> float | None:
        """Return the distance the table was made at (Mpc), if given."""
        return self._distance

    def pdescs(self) -> dict[str, ParamDesc]:
        return dict(upsilon=ParamScalarDesc(
            desc="the ratio of its mass to that of its table (e.g. a "
                 "mass-to-light ratio)", unit=''))

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
            scale = kpc_per_arcsec / (
                self._distance * u.Mpc * u.arcsec).to_value(
                    u.kpc, u.dimensionless_angles())
        if angles:
            return arcsec / self._radii.unit.to(u.arcsec), scale
        # (the lengths of the table, at D0, of those angles)
        lengths = (arcsec * u.arcsec * self._distance * u.Mpc).to(
            self._radii.unit, u.dimensionless_angles())
        return lengths.value, scale
