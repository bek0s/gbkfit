"""
The exact areas of the overlaps of shapes with the pixels of a grid, in
pixel coordinates: pixel (i, j) is the unit square centred on (i, j).
"""

import numpy as np


__all__ = [
    'ellipse_pixel_overlaps',
    'polygon_area',
    'polygon_pixel_overlaps'
]


def polygon_area(vertices: np.ndarray) -> float:
    """
    The signed area of a polygon (vertices of shape (n, 2)): positive if
    its vertices go counterclockwise.
    """
    x, y = np.asarray(vertices, float).T
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygon_pixel_overlaps(
        vertices: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """
    The areas of the overlaps of a simple polygon (vertices of shape
    (n, 2), in either order; it can be concave) with the pixels (i, j)
    (integer arrays of one shape). The area of a polygon in a pixel is
    that below and to the left of the pixel's top right corner, less those
    below and to the left of its top left and bottom right corners, plus
    that below and to the left of its bottom left corner.
    """
    vertices = np.asarray(vertices, float)
    i = np.asarray(i, float)
    j = np.asarray(j, float)
    return (
        _polygon_area_below_left(vertices, i + 0.5, j + 0.5)
        - _polygon_area_below_left(vertices, i - 0.5, j + 0.5)
        - _polygon_area_below_left(vertices, i + 0.5, j - 0.5)
        + _polygon_area_below_left(vertices, i - 0.5, j - 0.5))


def _polygon_area_below_left(vertices, x, y):
    """
    The area of the part of a polygon below y and to the left of x
    (arrays of one shape). Along a vertical line, the length of the
    polygon below y is the sum, over the edges the line crosses, of
    min(edge height, y), with the sign of the edge's direction (Green's
    theorem: the area of a counterclockwise polygon is minus the integral
    of y dx along its edges). So the area is minus the sum over the edges
    of the integral of min(edge height, y) dx, up to x.
    """
    x1, y1 = vertices.T
    x2, y2 = np.roll(vertices, -1, axis=0).T
    x = x[..., None]
    y = y[..., None]
    # The part of each edge left of x, from u to w (none for vertical
    # edges and for those right of x)
    u = np.minimum(x1, x2)
    w = np.clip(x, u, np.maximum(x1, x2))
    length = w - u
    slope = np.divide(
        y2 - y1, x2 - x1, out=np.zeros_like(y1), where=x2 != x1)
    height_u = y1 + (u - x1) * slope
    height_w = y1 + (w - x1) * slope
    # The integral of min(height, y) is that of the height less that of
    # max(height - y, 0), whose mean from u to w is excess (of a linear
    # height: the mean of the positive part of a linear function)
    a = height_u - y
    b = height_w - y
    a_pos = np.maximum(a, 0)
    b_pos = np.maximum(b, 0)
    excess = np.where(
        a == b, a_pos,
        np.divide(a_pos ** 2 - b_pos ** 2, 2 * (a - b),
                  out=np.zeros_like(a), where=a != b))
    integral = length * ((height_u + height_w) / 2 - excess)
    signed = np.sign(x2 - x1) * integral
    orientation = np.sign(polygon_area(vertices))
    return -orientation * np.sum(signed, axis=-1)


def ellipse_pixel_overlaps(
        matrix: np.ndarray, offset: np.ndarray,
        i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """
    The areas of the overlaps of the ellipse of the points p with
    |matrix @ p + offset| <= 1 (matrix of shape (2, 2), not singular) with
    the pixels (i, j) (integer arrays of one shape). The map takes the
    ellipse to the unit disk and each pixel to a parallelogram; the area
    of the disk inside a parallelogram, a sum over its edges, is divided
    by the area of the parallelogram (|det matrix|).
    """
    matrix = np.asarray(matrix, float)
    offset = np.asarray(offset, float)
    i = np.asarray(i, float)[..., None]
    j = np.asarray(j, float)[..., None]
    # The corners of each pixel, counterclockwise, then in disk coordinates
    corners_x = i + np.array([-0.5, 0.5, 0.5, -0.5])
    corners_y = j + np.array([-0.5, -0.5, 0.5, 0.5])
    corners = np.stack([corners_x, corners_y], axis=-1)
    corners = corners @ matrix.T + offset
    starts = corners
    ends = np.roll(corners, -1, axis=-2)
    area = np.sum(_unit_disk_triangle_area(starts, ends), axis=-1)
    return np.abs(area) / abs(np.linalg.det(matrix))


def _unit_disk_triangle_area(p, q):
    """
    The signed area of the part of the unit disk inside the triangle of
    the origin and the points p and q (arrays of shape (..., 2)), positive
    if the triangle goes counterclockwise. The segment from p to q is
    split where it crosses the circle: its pieces inside the disk add
    triangles with the origin, and those outside add sectors.
    """
    d = q - p
    qa = np.sum(d * d, axis=-1)
    qb = 2 * np.sum(p * d, axis=-1)
    qc = np.sum(p * p, axis=-1) - 1
    disc = qb ** 2 - 4 * qa * qc
    root = np.sqrt(np.maximum(disc, 0))
    # The segment is inside the circle between t1 and t2 (none if the
    # line misses the circle)
    t1 = np.where(disc > 0, np.clip((-qb - root) / (2 * qa), 0, 1), 1)
    t2 = np.where(disc > 0, np.clip((-qb + root) / (2 * qa), 0, 1), 1)
    p1 = p + t1[..., None] * d
    p2 = p + t2[..., None] * d
    return _sector_area(p, p1) + _cross(p1, p2) / 2 + _sector_area(p2, q)


def _cross(u, v):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _sector_area(u, v):
    """The signed area of the sector of the unit disk from u to v."""
    return np.arctan2(_cross(u, v), np.sum(u * v, axis=-1)) / 2
