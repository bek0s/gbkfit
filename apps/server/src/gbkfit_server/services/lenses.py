"""
The lenses of the projects, as their screen shows them: a lens of the
options of a piece described (its parameters, by the names of the lens,
and its problems), and what it makes of a source behind it (its
deflection over a field of its plane, its critical curves and its
caustics).
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

import contourpy
import numpy as np

from gbkfit.config import ConfigError, error_message
from gbkfit.cosmology import DEFAULT
from gbkfit.driver.drivers.host import DriverHost
from gbkfit.lensing import Deflects, Lens, SkyPoints, quoted_factor
from gbkfit.params import ParamSpace

from .errors import Invalid
from .workspace import Workspace


# The field of the view of a lens without strengths to size it (its
# half-width, arcsec), and the field of one with, in Einstein radii
_FIELD = 2.0
_FIELD_IN_EINSTEIN_RADII = 2.5


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
    out (the masses of galaxies, which their setups lay out).
    """
    half: float
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
            half: float | None = None
    ) -> LensView:
        """
        Return what a lens of the options of a piece of a project makes of
        a source behind it (see LensView), at the values of the properties
        of its parameters (see ParamSpace), over a field of the given
        half-width (by default, of its Einstein radii).

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
        if half is None:
            radii = [v for k, v in values.items() if k.endswith('theta_e')]
            half = _FIELD_IN_EINSTEIN_RADII * max(radii) if radii else _FIELD
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

        axis = np.linspace(-half, half, size)
        x, y = np.meshgrid(axis, axis)
        ax, ay = deflection(x.ravel(), y.ravel())
        ax, ay = ax.reshape(x.shape), ay.reshape(x.shape)
        # (the critical curves: where the determinant of the Jacobian of
        # the lens mapping, beta = theta - alpha, is 0; the caustics, those
        # points mapped to the source plane)
        step = axis[1] - axis[0]
        daxdy, daxdx = np.gradient(ax, step)
        daydy, daydx = np.gradient(ay, step)
        det = (1 - daxdx) * (1 - daydy) - daxdy * daydx
        lines = contourpy.contour_generator(axis, axis, det).lines(0.0)
        critical, caustics = [], []
        for line in lines:
            if len(line) < 3:
                continue
            bx, by = deflection(line[:, 0], line[:, 1])
            critical.append(_points(line[:, 0], line[:, 1]))
            caustics.append(_points(line[:, 0] - bx, line[:, 1] - by))
        return LensView(
            float(half), size, _rounded(ax), _rounded(ay), critical, caustics,
            [piece.name() for piece in made.pieces()
             if not isinstance(piece, Deflects)])


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
