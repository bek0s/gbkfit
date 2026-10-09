import abc
from collections.abc import Sequence
from numbers import Real

import numpy as np

from gbkfit.math import overlap
from gbkfit.utils import gridutils, parseutils


__all__ = [
    'Aperture',
    'ApertureCircle',
    'ApertureEllipse',
    'ApertureField',
    'AperturePolygon',
    'ApertureRectangle',
    'aperture_parser'
]


# The smallest overlap of a pixel with an aperture that counts, as a
# fraction of the pixel (smaller ones are rounding errors of zero)
_MIN_OVERLAP = 1e-12

# The largest relative difference between the area of an aperture and
# the area of its pixels on a grid that covers it (rounding errors)
_AREA_TOLERANCE = 1e-9


def _axes(posa):
    """
    The unit vectors on the sky along a position angle (degrees, north
    through east, like posa) and across it (90 degrees clockwise).
    """
    posa = np.radians(posa)
    along = np.array([-np.sin(posa), np.cos(posa)])
    across = np.array([np.cos(posa), np.sin(posa)])
    return along, across


def _pixels_in_box(grid, lower, upper):
    """
    The pixel indices (i, j) of a spatial grid whose pixels overlap the box
    from lower to upper (pixel coordinates), and the flat indices of the
    pixels.
    """
    size_x, size_y = grid.size[:2]
    i_min = max(0, int(np.floor(lower[0] + 0.5)))
    i_max = min(size_x - 1, int(np.ceil(upper[0] - 0.5)))
    j_min = max(0, int(np.floor(lower[1] + 0.5)))
    j_max = min(size_y - 1, int(np.ceil(upper[1] - 0.5)))
    # (none, for a box outside the grid)
    i_max = max(i_max, i_min - 1)
    j_max = max(j_max, j_min - 1)
    j, i = np.mgrid[j_min:j_max + 1, i_min:i_max + 1]
    return i.ravel(), j.ravel(), (j * size_x + i).ravel()


def _overlaps_inside_grid(aperture, indices, areas, area):
    """
    The pixels (flat indices) that overlap an aperture and the fraction of
    each inside it, given the areas of the overlaps (pixel units) of the
    pixels of a grid. Raise RuntimeError if the aperture, of the given
    area, is not inside the grid.
    """
    inside = np.sum(areas)
    if inside < area * (1 - _AREA_TOLERANCE):
        desc = parseutils.make_typed_desc(aperture.__class__, 'aperture')
        raise RuntimeError(
            f"{desc} is not inside the grid of the model "
            f"({inside / area:.1%} of it is); enlarge the grid")
    keep = areas > _MIN_OVERLAP
    return indices[keep], areas[keep]


class Aperture(parseutils.TypedSerializable, abc.ABC):
    """
    A region of the sky whose data are the sum of the pixels of a grid,
    each weighted by the fraction of its area inside the region (exactly).
    Positions are in arcsec in the frame of the model (x and y from the
    reference pixel of the grid, like xpos and ypos), position angles in
    degrees, north through east (like posa).
    """

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'aperture')
        return cls(**parseutils.parse_options_for_callable(
            info, desc, cls.__init__))

    @abc.abstractmethod
    def overlaps(self, grid: gridutils.Grid) -> tuple[np.ndarray, np.ndarray]:
        """
        The pixels of a spatial grid (flat indices, x fastest) that overlap
        the aperture, and the fraction of the area of each inside it.
        Raise RuntimeError unless the aperture is inside the grid.
        """
        pass


class ApertureEllipse(Aperture):
    """An ellipse of semi-axes a (along posa) and b."""

    @staticmethod
    def type():
        return 'ellipse'

    def dump(self):
        return dict(
            type=self.type(), x=self._x, y=self._y, a=self._a, b=self._b,
            posa=self._posa)

    def __init__(self, x: Real, y: Real, a: Real, b: Real, posa: Real = 0):
        if not (a > 0 and b > 0):
            raise RuntimeError(
                f"the semi-axes of an ellipse must be positive; they are "
                f"{a} and {b}")
        self._x = x
        self._y = y
        self._a = a
        self._b = b
        self._posa = posa

    def overlaps(self, grid):
        return _ellipse_overlaps(
            self, grid, (self._x, self._y), self._a, self._b, self._posa)


class ApertureCircle(Aperture):
    """A circle."""

    @staticmethod
    def type():
        return 'circle'

    def dump(self):
        return dict(type=self.type(), x=self._x, y=self._y, radius=self._radius)

    def __init__(self, x: Real, y: Real, radius: Real):
        if not radius > 0:
            raise RuntimeError(
                f"the radius of a circle must be positive; it is {radius}")
        self._x = x
        self._y = y
        self._radius = radius

    def overlaps(self, grid):
        return _ellipse_overlaps(
            self, grid, (self._x, self._y), self._radius, self._radius, 0)


def _ellipse_overlaps(aperture, grid, centre, a, b, posa):
    """
    The overlaps of an ellipse aperture with the pixels of a grid (see
    Aperture.overlaps). The map from pixels to the coordinates in which
    the ellipse is the unit disk: disk = ellipse @ (sky - centre), with
    sky = inverse(matrix) @ (pixel - offset).
    """
    along, across = _axes(posa)
    ellipse = np.array([along / a, across / b])
    matrix, offset = gridutils.sky_to_pixel(grid)
    inverse = np.linalg.inv(matrix)
    disk_matrix = ellipse @ inverse
    disk_offset = -ellipse @ (inverse @ offset + np.asarray(centre, float))
    # The box of the ellipse in pixel coordinates
    pixel_centre = np.linalg.solve(disk_matrix, -disk_offset)
    half_size = np.linalg.norm(np.linalg.inv(disk_matrix), axis=1)
    i, j, indices = _pixels_in_box(
        grid, pixel_centre - half_size, pixel_centre + half_size)
    areas = overlap.ellipse_pixel_overlaps(disk_matrix, disk_offset, i, j)
    area = np.pi / abs(np.linalg.det(disk_matrix))
    return _overlaps_inside_grid(aperture, indices, areas, area)


class AperturePolygon(Aperture):
    """A simple polygon (it can be concave), of the given vertices."""

    @staticmethod
    def type():
        return 'polygon'

    def dump(self):
        return dict(type=self.type(), vertices=self._vertices.tolist())

    def __init__(self, vertices: Sequence[Sequence[Real]]):
        vertices = np.array(vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2 \
                or vertices.shape[0] < 3:
            raise RuntimeError(
                f"a polygon needs at least three vertices, each with x and "
                f"y; its vertices have the shape {vertices.shape}")
        if overlap.polygon_area(vertices) == 0:
            raise RuntimeError("the polygon has no area")
        self._vertices = vertices

    def overlaps(self, grid):
        return _polygon_overlaps(self, grid, self._vertices)


class ApertureRectangle(Aperture):
    """A rectangle, its length along posa (e.g. a slit or a shutter)."""

    @staticmethod
    def type():
        return 'rectangle'

    def dump(self):
        return dict(
            type=self.type(), x=self._x, y=self._y, length=self._length,
            width=self._width, posa=self._posa)

    def __init__(
            self, x: Real, y: Real, length: Real, width: Real,
            posa: Real = 0):
        if not (length > 0 and width > 0):
            raise RuntimeError(
                f"the length and the width of a rectangle must be positive; "
                f"they are {length} and {width}")
        self._x = x
        self._y = y
        self._length = length
        self._width = width
        self._posa = posa

    def overlaps(self, grid):
        along, across = _axes(self._posa)
        half_along = along * self._length / 2
        half_across = across * self._width / 2
        centre = np.array([self._x, self._y], dtype=float)
        vertices = np.array([
            centre - half_along - half_across,
            centre - half_along + half_across,
            centre + half_along + half_across,
            centre + half_along - half_across])
        return _polygon_overlaps(self, grid, vertices)


def _polygon_overlaps(aperture, grid, vertices):
    """
    The overlaps of a polygon aperture (vertices on the sky) with the
    pixels of a grid (see Aperture.overlaps).
    """
    matrix, offset = gridutils.sky_to_pixel(grid)
    pixel_vertices = vertices @ matrix.T + offset
    i, j, indices = _pixels_in_box(
        grid, pixel_vertices.min(axis=0), pixel_vertices.max(axis=0))
    areas = overlap.polygon_pixel_overlaps(pixel_vertices, i, j)
    area = abs(overlap.polygon_area(pixel_vertices))
    return _overlaps_inside_grid(aperture, indices, areas, area)


class ApertureField(Aperture):
    """The whole grid of the model (e.g. for an integrated spectrum)."""

    @staticmethod
    def type():
        return 'field'

    def dump(self):
        return dict(type=self.type())

    def __init__(self):
        pass

    def overlaps(self, grid):
        npix = int(np.prod(grid.size[:2]))
        return np.arange(npix), np.ones(npix)


aperture_parser = parseutils.TypedParser(Aperture, [
    ApertureCircle,
    ApertureEllipse,
    ApertureField,
    AperturePolygon,
    ApertureRectangle])
