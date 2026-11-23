"""
The lenses of the projects, as their screen shows them: a lens of the
options of a piece described (its parameters, by the names of the lens,
and its problems), and what it makes of a source behind it (its
deflection over a field of its plane, its critical curves and its
caustics).
"""

import dataclasses
import math
from collections.abc import Callable, Mapping
from typing import Any

import contourpy
import numpy as np

from gbkfit.config import ConfigError, error_message
from gbkfit.cosmology import DEFAULT
from gbkfit.driver.drivers.host import DriverHost
from gbkfit.lensing import (
    Deflects, Lens, LensEPL, LensNFW, LensSIE, SkyPoints, quoted_factor)
from gbkfit.params import ModelEvaluationError, ParamSpace

from .errors import Invalid
from .workspace import Workspace


# The field a view of a lens is fitted to (see _fitted_half): its
# half-width, of a lens of no Einstein radius (arcsec), and of one with,
# in Einstein radii; and the round fields (their width, arcsec, times a
# power of ten; the Studio zooms by them too)
_FIELD = 2.0
_FIELD_IN_EINSTEIN_RADII = 2.5
_ROUND_FIELDS = (1, 1.5, 2, 3, 4, 6, 8)

# The rays the Einstein radius of a lens is found along (see
# _einstein_radius): their number, and the distances along them (arcsec)
_RAYS = 16
_RADII = np.geomspace(1e-3, 1e3, 241)

# The pieces placed by a centre (see the module of the pieces)
_CENTRED = (LensSIE, LensEPL, LensNFW)


@dataclasses.dataclass
class LensDescription:
    """
    A lens described: its parameters, by the names of the lens (e.g.
    sie_theta_e; see Lens.pdescs), as those of a galaxy (see
    GalaxyDescription; their owner is their piece, ['pieces', i]), and its
    problems (none but those of the lens: the values of its parameters are
    not given).
    """
    parameters: list[dict[str, Any]]
    problems: list[dict[str, Any]]


@dataclasses.dataclass
class LensView:
    """
    What a lens makes of a source behind it, at the source_redshift of the
    lens, or else of its first quoted piece (or at D_ls / D_s = 1 without
    quoted pieces; see _view_source): the field of
    the view (half its width, arcsec) and the number of points across, the
    deflection at each point of the plane (arcsec, row-major, y up), its
    critical curves and its caustics (lists of points), and the pieces left
    out (the masses of galaxies, which their setups lay out); the field is
    of a half-width and about a centre (x and y, arcsec).
    """
    half: float
    centre: tuple[float, float]
    size: int
    alpha_x: list[float]
    alpha_y: list[float]
    critical: list[list[list[float]]]
    caustics: list[list[list[float]]]
    left_out: list[str]


class Lenses:
    """The lenses of the projects of a workspace."""

    def __init__(self, workspace: Workspace):
        self._workspace = workspace

    def describe(
            self, project: str, lens: Mapping[str, Any]
    ) -> LensDescription:
        """
        Describe a lens of the options of a piece of a project (see
        LensDescription).
        """
        try:
            made = _made(lens)
        except (ConfigError, ValueError, TypeError) as e:
            return LensDescription([], [dict(
                severity='error', path=list(getattr(e, 'path', []) or []),
                message=error_message(e))])
        owners = _owners(made)
        return LensDescription([
            dict(name=name, title=desc.title(), desc=desc.desc(),
                 size=desc.size(), default=desc.default(),
                 unit=None if desc.unit() is None else str(desc.unit()),
                 range=str(desc.valid_range()), period=desc.period(),
                 owner=['pieces', owners[name]])
            for name, desc in made.pdescs().items()], [])

    def view(
            self,
            project: str,
            lens: Mapping[str, Any],
            properties: Mapping[str, Any],
            size: int,
            half: float | None = None,
            centre: tuple[float, float] | None = None
    ) -> LensView:
        """
        Return what a lens of the options of a piece of a project makes of
        a source behind it (see LensView), at the values of the properties
        of its parameters (see ParamSpace), over a field of the given
        half-width and centre (by default, fitted to the lens: see
        _fitted_centre and _fitted_half).

        Raises
        ------
        Invalid
            If the lens or the properties are invalid.
        """
        try:
            made = _made(lens)
            values = ParamSpace.from_properties(
                made.pdescs(), dict(properties)).evaluate()
        except (ConfigError, ValueError, TypeError) as e:
            raise Invalid(error_message(e)) from None
        values = {
            name: float(np.ravel(values[name])[0]) for name in made.pdescs()}
        factors = made.quoted_factors(made.redshift(), DEFAULT)
        source = _view_source(made)
        view = 1.0 if source is None \
            else quoted_factor(made.redshift(), source, DEFAULT)
        driver = DriverHost()

        def deflection(
                x: np.ndarray, y: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            # (at the source of the view: alpha-hat D_ls / D_s)
            ax, ay = made.deflection(
                SkyPoints(x, y, (0.0, 0.0)), values, factors, driver)
            return np.asarray(ax) / view, np.asarray(ay) / view

        if centre is None:
            centre = _fitted_centre(made, values)
        if half is None:
            half = _fitted_half(made, values, centre, deflection)
        axis_x = np.linspace(centre[0] - half, centre[0] + half, size)
        axis_y = np.linspace(centre[1] - half, centre[1] + half, size)
        x, y = np.meshgrid(axis_x, axis_y)
        ax, ay = deflection(x.ravel(), y.ravel())
        ax, ay = ax.reshape(x.shape), ay.reshape(x.shape)
        # (the critical curves: where the determinant of the Jacobian of
        # the lens mapping, beta = theta - alpha, is 0; the caustics, those
        # points mapped to the source plane)
        step = axis_x[1] - axis_x[0]
        daxdy, daxdx = np.gradient(ax, step)
        daydy, daydx = np.gradient(ay, step)
        det = (1 - daxdx) * (1 - daydy) - daxdy * daydx
        lines = contourpy.contour_generator(axis_x, axis_y, det).lines(0.0)
        critical, caustics = [], []
        for line in lines:
            if len(line) < 3:
                continue
            bx, by = deflection(line[:, 0], line[:, 1])
            critical.append(_points(line[:, 0], line[:, 1]))
            caustics.append(_points(line[:, 0] - bx, line[:, 1] - by))
        return LensView(
            float(half), (float(centre[0]), float(centre[1])), size,
            _rounded(ax), _rounded(ay), critical, caustics,
            [piece.name() for piece in made.pieces()
             if not isinstance(piece, Deflects)])


def _centres(
        lens: Lens, values: Mapping[str, float]
) -> list[tuple[float, float]]:
    """Return the centres of the pieces of a lens placed by one."""
    return [
        (values[f'{piece.name()}_xpos'], values[f'{piece.name()}_ypos'])
        for piece in lens.pieces() if isinstance(piece, _CENTRED)]


def _fitted_centre(
        lens: Lens, values: Mapping[str, float]
) -> tuple[float, float]:
    """
    Return the centre of the field a view of a lens is fitted to: the
    middle of the centres of its pieces, (0, 0) if none has one (that of a
    shear and of a sheet).
    """
    centres = _centres(lens, values)
    if not centres:
        return 0.0, 0.0
    xs, ys = zip(*centres)
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def _fitted_half(
        lens: Lens,
        values: Mapping[str, float],
        centre: tuple[float, float],
        deflection: Callable[
            [np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
) -> float:
    """
    Return the half-width of the field a view of a lens about a centre is
    fitted to: a round field (see _ROUND_FIELDS) that holds its Einstein
    radius times _FIELD_IN_EINSTEIN_RADII (or _FIELD, without one) and the
    centres of its pieces, and that its deflection covers.
    """
    radius, reach = _einstein_radius(centre, deflection)
    half = _FIELD if radius is None else _FIELD_IN_EINSTEIN_RADII * radius
    for x, y in _centres(lens, values):
        half = max(half, 1.25 * math.hypot(x - centre[0], y - centre[1]))
    # (the field, whose corners are within the reach of the deflection)
    largest = 2 * reach / math.sqrt(2)
    field = _round_field(2 * half, up=True)
    if field > largest:
        field = _round_field(largest, up=False)
    return field / 2


def _einstein_radius(
        centre: tuple[float, float],
        deflection: Callable[
            [np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
) -> tuple[float | None, float]:
    """
    Return the Einstein radius of a lens about a centre, the largest along
    the rays from it (where its deflection outwards falls short of the
    distance, as at the ring of a round lens), None if it has none (e.g. a
    shear, a sheet), and the reach of its deflection: the distance it is
    known within (deflection maps cover a part of the sky).
    """
    angles = np.linspace(0, 2 * np.pi, _RAYS, endpoint=False)[:, None]
    radii = _RADII
    while True:
        x = centre[0] + radii * np.cos(angles)
        y = centre[1] + radii * np.sin(angles)
        try:
            ax, ay = deflection(x.ravel(), y.ravel())
            break
        except ModelEvaluationError:
            # (closer, by halves)
            radii = radii[radii <= radii[-1] / 2]
            if len(radii) < 2:
                return None, float(_RADII[0])
    outwards = (ax.reshape(x.shape) * np.cos(angles)
                + ay.reshape(y.shape) * np.sin(angles))
    # (along each ray, the distance past the last that the deflection
    # reaches beyond, of the rays it falls short of in the end)
    reaches = outwards > radii
    found = [
        radii[min(np.flatnonzero(ray)[-1] + 1, len(radii) - 1)]
        for ray in reaches if ray.any() and not ray[-1]]
    return (float(max(found)) if found else None), float(radii[-1])


def _round_field(field: float, up: bool) -> float:
    """
    Return the round field (see _ROUND_FIELDS) nearest a field, of those
    not smaller (up) or not larger.
    """
    power = 10.0 ** math.floor(math.log10(field))
    rounds = [r * power for r in (*_ROUND_FIELDS, 10)]
    if up:
        return next(r for r in rounds if r >= field * (1 - 1e-9))
    return max(r for r in rounds if r <= field * (1 + 1e-9))


def _made(lens: Mapping[str, Any]) -> Lens:
    """Return the lens of options."""
    return Lens.load(dict(lens))


def _owners(lens: Lens) -> dict[str, int]:
    """Return the piece of each parameter of a lens (its index), by name."""
    owners = {}
    for i, piece in enumerate(lens.pieces()):
        for name in piece.pdescs():
            owners[f'{piece.name()}_{name}'] = i
    return owners


def _points(x: np.ndarray, y: np.ndarray) -> list[list[float]]:
    """Return points as pairs, rounded."""
    return [[round(float(a), 5), round(float(b), 5)] for a, b in zip(x, y)]


def _rounded(values: np.ndarray) -> list[float]:
    """Return the values of a map, row-major, rounded."""
    return [round(float(v), 5) for v in values.ravel()]


def _view_source(lens: Lens) -> float | None:
    """
    Return the redshift of the source a view of a lens is of: its own
    source_redshift, or else that of its first quoted piece; None without
    quoted pieces (a view at D_ls / D_s = 1). Each piece is scaled to it
    from its own source.
    """
    if (source := lens.source_redshift()) is not None:
        return source
    return next((
        source for piece in lens.pieces()
        if (source := lens.source_redshift_of(piece)) is not None), None)
