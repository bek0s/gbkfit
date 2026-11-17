"""
The mass of a galaxy laid out on the sky (SkyMass), for the lenses of
its mass: a closed set of plain layers of projected mass (points, round
layers, elliptical layers and bands of rings), in kpc and Msun, which
lensing turns into deflections. The mass components that can be laid
out on the sky give their layers (see SkyProjected), placed by the
components of their galaxy (see SkyPlacement).

The layers are on the plane of the sky, in the frame of the parameters
of the galaxies (+x to the west, +y to the north, from the origin of
their xpos and ypos), in kpc: the arcsec of the frame times the length
of an arcsec in the galaxy (see SkyMass.kpc_per_arcsec).

Each layer is a factor (scale) times a unit shape: the mass of the layer
is scale times that of its shape. A component puts in scale what one of
its parameters scales alone (e.g. the mass of a point, or the mass per
unit of the light of a component it traces), so that the unit shape,
and its deflection, stay the same while only that parameter changes
(see same_shape).

The elliptical layers and the bands are sheets of an axisymmetric
surface density Sigma(R), R the radius in the plane of the sheet,
stretched onto the sky by sigma1 along the position angle posa and by
sigma2 across it (sigma1 >= sigma2): a face-on disk seen at the
inclination i is stretched by 1 and cos(i). A thick layer is a sum of
such sheets (slices) at heights h along the normal of its plane (a
vector of the sky, of length sin(i)).
"""

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple, Protocol, TypeAlias, \
    runtime_checkable

import numpy as np
import numpy.typing as npt

from gbkfit.params import ModelEvaluationError, ParamValues
from . import _numerics

if TYPE_CHECKING:
    from .base import MassContext


__all__ = [
    'BandLayer',
    'EllipticalLayer',
    'LinearBetweenKnots',
    'LogTable',
    'MassPlane',
    'PointLayer',
    'RingGeometry',
    'RingPlacement',
    'RoundLayer',
    'SkyLayer',
    'SkyMass',
    'SkyPlacement',
    'SkyProjected',
    'SpheroidAxis',
    'check_planes',
    'plane_normal',
    'plane_tilt',
    'projected_ratio'
]


# The radii (kpc) of the tables of the layers (see LogTable): from 1e-6
# kpc (a milliparsec, far inside any ray) to 1e5 kpc (where no galaxy has
# mass left that bends light near it), 64 per decade, on which the cubic
# interpolation of the tables is good to about 1e-7
_TABLE_START = math.log(1e-6)
_TABLE_STEP = math.log(10) / 64
_TABLE_SIZE = 11 * 64 + 1

# The tail of the tables of surface densities that the layers drop: where
# Sigma R^2, the mass per unit of ln R, is below this share of its largest
# value (the deflection of a ray inside an elliptical shell does not see
# it, and a ray outside one sees its mass, here negligible)
_TAIL = 1e-16

# The slices of a thick layer (see height_slices): Gauss-Legendre nodes
# on each of the panels between these heights, in units of its scale
# height, on either side of its plane: 84 slices, uniform to 3 scale
# heights (a thin sheet seen near edge-on bends light sharply across it,
# and the slices must be as close everywhere the layer has mass; a
# geometric layout of 48 slices was 4 times further off at 85 degrees)
_HEIGHT_PANELS = (
    0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4,
    5, 6, 8, 10, 12, 16, 24)
_HEIGHT_NODES = 2

# The most edge-on a thick layer is laid out in slices (see
# check_slices): its median height times the tangent of its inclination
# over its scale length. Measured on exponential disks (sech2 and
# exponential in height, hz/rd 0.13 and 0.33, at z 0.35, rays 0.1 to 1
# arcsec from the centre): at this ratio, the deflection near the centre
# is off by about 1e-3 of itself (0.1 mas for a disk of 5e10 Msun), and
# beyond, it grows as the square of the ratio
_EDGE_ON_RATIO = 1.5

# The largest angle (degrees) between the planes of the components that
# place the mass in one plane (see SkyPlacement.plane): planes tied or
# fixed at the same values agree to the last bits
_PLANE_TOLERANCE = 1e-9


@dataclasses.dataclass(frozen=True, eq=False)
class LogTable:
    """
    A positive radial function on geometric radii (e.g. those of
    table_radii): its logarithm at the radii exp(start + k step), k from
    0, cubic between them in ln R (4-point Lagrange, of the stencil k - 1
    to k + 2 of the interval k, clamped to the table); below the first and
    beyond the last, the power laws of the two first and the two last. A
    table of several functions (e.g. of the slices of a layer) has one row
    of values for each. The native kernels interpolate it as here (see
    gbkfit/lensing/layers.hpp).

    Attributes
    ----------
    start : float
        The logarithm of the first radius (in its unit of length or angle;
        kpc of the layers).
    step : float
        The step of the logarithm of the radii.
    values : np.ndarray
        The logarithm of the function at each radius: of shape (n,), or
        (rows, n).
    """
    start: float
    step: float
    values: np.ndarray

    def __call__(self, R: npt.ArrayLike) -> np.ndarray:
        """
        Return the function at the radii R (positive, in its unit): of the
        shape of R, or of a table of several, (rows,) + R.shape.
        """
        R = np.asarray(R, dtype=float)
        n = self.values.shape[-1]
        x = (np.log(R) - self.start) / self.step
        index = np.clip(np.floor(x).astype(int), 1, n - 3)
        t = x - index
        weights = (-t * (t - 1) * (t - 2) / 6, (t + 1) * (t - 1) * (t - 2) / 2,
                   -(t + 1) * t * (t - 2) / 2, (t + 1) * t * (t - 1) / 6)

        def at(k: npt.ArrayLike) -> np.ndarray:
            # (the values, logarithms, at the indices k, of the shape of R)
            return np.take(self.values, np.broadcast_to(k, x.shape), axis=-1)
        cubic = sum(w * at(index + k - 1) for k, w in enumerate(weights))
        # (beyond the ends: the power laws of the two last radii)
        first = at(0) + x * (at(1) - at(0))
        last = at(n - 1) + (x - n + 1) * (at(n - 1) - at(n - 2))
        return np.exp(np.where(x < 0, first, np.where(x > n - 1, last, cubic)))


def table_radii() -> np.ndarray:
    """
    Return the radii of the tables of the layers (see LogTable).

    Returns
    -------
    np.ndarray
        The radii (kpc): geometric, from 1e-6 to 1e5 kpc, 64 per decade.
    """
    return np.exp(_TABLE_START + _TABLE_STEP * np.arange(_TABLE_SIZE))


def _log(values: np.ndarray) -> np.ndarray:
    """
    Return the logarithm of the values of a table, its zeros (e.g. of a
    layer of no mass) at the logarithm of the smallest positive float.
    """
    return np.log(np.maximum(values, np.finfo(float).tiny))


def _same(first: object, second: object) -> bool:
    """
    Check whether two layers of a kind have the same unit shape: every
    field equal to the last bits but scale.
    """
    if type(first) is not type(second):
        return False
    for field in dataclasses.fields(first):
        if field.name == 'scale':
            continue
        a, b = getattr(first, field.name), getattr(second, field.name)
        if isinstance(a, LogTable):
            if not (a.start == b.start and a.step == b.step
                    and np.array_equal(a.values, b.values)):
                return False
        elif not np.array_equal(a, b):
            return False
    return True


@dataclasses.dataclass(frozen=True, eq=False)
class PointLayer:
    """
    A point mass: of scale Msun, at its centre.

    Attributes
    ----------
    centre : tuple of float
        Its position (kpc, see the module).
    scale : float
        Its mass (Msun).
    """
    centre: tuple[float, float]
    scale: float

    def same_shape(self, other: 'SkyLayer') -> bool:
        """Check whether another layer has its unit shape (see _same)."""
        return _same(self, other)


@dataclasses.dataclass(frozen=True, eq=False)
class RoundLayer:
    """
    A round layer: a circular surface density about its centre, given by
    its projected mass M2D(<R) within the circles of the radii R about it:
    scale times its table (Msun per unit scale).

    Attributes
    ----------
    centre : tuple of float
        Its centre (kpc, see the module).
    scale : float
        Its scale (see the module).
    mass : LogTable
        Its projected mass within each radius per unit scale (Msun).
    """
    centre: tuple[float, float]
    scale: float
    mass: LogTable

    def same_shape(self, other: 'SkyLayer') -> bool:
        """Check whether another layer has its unit shape (see _same)."""
        return _same(self, other)


@dataclasses.dataclass(frozen=True, eq=False)
class EllipticalLayer:
    """
    An elliptical layer: a sheet of a smooth surface density Sigma(R) in
    its plane, stretched onto the sky (see the module), in slices: scale
    times the table of each slice (Msun/kpc^2 per unit scale, in its
    plane), at its height along the normal of its plane, the slices
    adding up. The surface density of the sky is Sigma(a / sigma1) /
    (sigma1 sigma2) on the ellipse of the semi-major axis a (along posa)
    and the axis ratio sigma2 / sigma1, and 0 beyond the last radius of
    its tables.

    Attributes
    ----------
    centre : tuple of float
        Its centre (kpc, see the module).
    scale : float
        Its scale (see the module).
    posa : float
        The position angle (degrees, north through east, as that of
        disks) of its major axis.
    stretch : tuple of float
        sigma1 and sigma2, sigma1 >= sigma2 >= 0 (see the module).
    sigma : LogTable
        The surface density of each slice per unit scale (Msun/kpc^2): a
        table of a row for each.
    heights : np.ndarray
        The height of each slice (kpc) along normal.
    normal : tuple of float
        The vector of the sky (kpc per kpc of height) of the normal of its
        plane.
    """
    centre: tuple[float, float]
    scale: float
    posa: float
    stretch: tuple[float, float]
    sigma: LogTable
    heights: np.ndarray
    normal: tuple[float, float]

    def same_shape(self, other: 'SkyLayer') -> bool:
        """Check whether another layer has its unit shape (see _same)."""
        return _same(self, other)


@dataclasses.dataclass(frozen=True, eq=False)
class BandLayer:
    """
    Bands of rings: sheets of a surface density that is a quadratic in
    the radius R within each band of radii (R_in <= R < R_out) and 0
    outside, each about its own centre, stretched onto the sky (see the
    module): the image of a ring of a disk, tilted and moved as the disk
    draws it (homoeoid bands). In slices, as the elliptical layers: scale
    times the quadratic of each band in each slice (Msun/kpc^2 per unit
    scale, in its plane), at the height of the slice along the normal of
    the band. The bands add up, overlapping or not.

    Attributes
    ----------
    scale : float
        Its scale (see the module).
    centres : np.ndarray
        The centre of each band (kpc, see the module): of shape (bands, 2).
    posa : np.ndarray
        The position angle (degrees) of the major axis of each band.
    stretch : np.ndarray
        sigma1 and sigma2 of each band: of shape (bands, 2).
    edges : np.ndarray
        R_in and R_out (kpc, in its plane) of each band: of shape (bands,
        2).
    density : np.ndarray
        The coefficients c0, c1, c2 of the surface density c0 + c1 R + c2
        R^2 of each band in each slice: of shape (slices, bands, 3).
    heights : np.ndarray
        The height of each slice (kpc).
    normals : np.ndarray
        The vector of the sky of the normal of the plane of each band: of
        shape (bands, 2).
    """
    scale: float
    centres: np.ndarray
    posa: np.ndarray
    stretch: np.ndarray
    edges: np.ndarray
    density: np.ndarray
    heights: np.ndarray
    normals: np.ndarray

    def same_shape(self, other: 'SkyLayer') -> bool:
        """Check whether another layer has its unit shape (see _same)."""
        return _same(self, other)


SkyLayer: TypeAlias = PointLayer | RoundLayer | EllipticalLayer | BandLayer


@dataclasses.dataclass(frozen=True, eq=False)
class SkyMass:
    """
    The mass of a galaxy laid out on the sky (see the module): the layers
    of each of its mass components, by the name of the component, and the
    length of an arcsec in the galaxy, which takes positions of the frame
    (arcsec) to those of the layers (kpc).

    Attributes
    ----------
    kpc_per_arcsec : float
        The length of an arcsec in the galaxy (kpc).
    parts : Mapping
        The layers of each mass component, by its name.
    """
    kpc_per_arcsec: float
    parts: Mapping[str, tuple[SkyLayer, ...]]

    def layers(self) -> tuple[SkyLayer, ...]:
        """Return the layers of all its mass components, in order."""
        return tuple(layer for part in self.parts.values() for layer in part)


class MassPlane(NamedTuple):
    """
    The plane of the mass of a galaxy (see SkyPlacement.plane): its
    position angle and inclination (degrees), as those of disks.
    """
    posa: float
    incl: float


class SpheroidAxis(NamedTuple):
    """
    Where a spheroid of a galaxy is: its centre (x, y; kpc, see the
    module), and the position angle and inclination (degrees) of its axis,
    None for a sphere.
    """
    x: float
    y: float
    posa: float | None
    incl: float | None


@dataclasses.dataclass(frozen=True, eq=False)
class RingGeometry:
    """
    Where rings of a disk of a galaxy are, as the disk draws them, at the
    radii asked (see RingPlacement): each ring's centre (x, y; kpc, see
    the module), the position angle and inclination (degrees) of its
    plane, and its vertical distortions (those of order 0 and 1, the
    displacement of each point of the ring along the normal of its plane:
    lift + tilt_x x_d + tilt_y y_d, x_d along its major axis and y_d 90
    degrees from it in its plane, as the coordinates of the plane of a
    disk): its lift (kpc) and the slopes of its tilt (kpc per kpc). The
    rings of a disk of distortions of higher orders are not ellipses on the
    sky, and have no geometry here.

    Attributes
    ----------
    x, y, posa, incl, lift, tilt_x, tilt_y : np.ndarray
        Of each ring (see above).
    """
    x: np.ndarray
    y: np.ndarray
    posa: np.ndarray
    incl: np.ndarray
    lift: np.ndarray
    tilt_x: np.ndarray
    tilt_y: np.ndarray


class RingPlacement(Protocol):
    """Where the rings of a disk of a galaxy are (see RingGeometry)."""

    def rings(self, radii: np.ndarray) -> RingGeometry:
        """
        Return the geometry of its rings at the radii (kpc, in their
        planes), as the disk draws them: of each, its parameters
        interpolated as the disk interpolates them, and beyond its first
        and last rings, those of its first and last.
        """
        ...


@dataclasses.dataclass(frozen=True, eq=False)
class SkyPlacement:
    """
    Where a galaxy places its mass on the sky: its one centre, and the
    components that its mass moves, holds up or follows (the disks by
    their rings, the spheroids by their axes), by name. A mass component
    that follows a component takes its geometry (see rings and axis); the
    others lie in the plane of the mass at its centre (see plane).

    Attributes
    ----------
    centre : tuple of float
        The centre of the mass (kpc, see the module).
    disks : Mapping
        The rings of each disk, by the name of its component.
    spheroids : Mapping
        The axis of each spheroid, by the name of its component.
    """
    centre: tuple[float, float]
    disks: Mapping[str, RingPlacement] = dataclasses.field(
        default_factory=dict)
    spheroids: Mapping[str, SpheroidAxis] = dataclasses.field(
        default_factory=dict)

    def rings(self, component: str, radii: np.ndarray) -> RingGeometry:
        """
        Return the geometry of the rings of the disk of a component at the
        radii (kpc; see RingPlacement).

        Raises
        ------
        ValueError
            If the placement has no disk of the name.
        """
        if component not in self.disks:
            raise ValueError(
                f"the mass follows the disk {component!r}, which does not "
                f"place it; the disks that do: {list(self.disks)}")
        return self.disks[component].rings(np.asarray(radii, dtype=float))

    def axis(self, component: str) -> SpheroidAxis:
        """
        Return the axis of the spheroid of a component.

        Raises
        ------
        ValueError
            If the placement has no spheroid of the name.
        """
        if component not in self.spheroids:
            raise ValueError(
                f"the mass follows the spheroid {component!r}, which does "
                f"not place it; the spheroids that do: "
                f"{list(self.spheroids)}")
        return self.spheroids[component]

    def plane(self) -> MassPlane:
        """
        Return the plane of the mass: that of the innermost ring of each
        disk and the axis of each spheroid that has one, which must agree
        (see check_planes; a disk that warps beyond its innermost ring does
        not move it).

        Raises
        ------
        ModelEvaluationError
            If they disagree, or there are none (no component gives the
            mass a plane).
        """
        planes = [
            (f"the disk {name!r}", float(rings.posa[0]), float(rings.incl[0]))
            for name, placement in self.disks.items()
            for rings in [placement.rings(np.zeros(1))]]
        planes += [
            (f"the spheroid {name!r}", axis.posa, axis.incl)
            for name, axis in self.spheroids.items()
            if axis.posa is not None]
        return check_planes(planes)


def plane_normal(posa: float, incl: float) -> np.ndarray:
    """
    Return the normal of a plane of the given position angle and
    inclination (degrees, as those of disks).

    Parameters
    ----------
    posa, incl : float
        The position angle and the inclination (degrees).

    Returns
    -------
    np.ndarray
        The unit normal in 3d: along the sky x and y (as the normals of
        layers, see the module) and the line of sight.
    """
    p, i = math.radians(posa), math.radians(incl)
    return np.array([
        math.sin(i) * math.cos(p), math.sin(i) * math.sin(p), math.cos(i)])


def plane_tilt(
        first: tuple[float, float], second: tuple[float, float]
) -> float:
    """
    Return the angle between two planes.

    Parameters
    ----------
    first, second : tuple of float
        The position angle and the inclination (degrees) of each.

    Returns
    -------
    float
        The angle (degrees, from 0 to 90) between them, atan2(|n1 x n2|,
        |n1 . n2|) of their normals (see plane_normal): exact for planes
        that agree but for the last bits (unlike the arc cosine of n1 .
        n2).
    """
    n1, n2 = plane_normal(*first), plane_normal(*second)
    return math.degrees(math.atan2(
        float(np.linalg.norm(np.cross(n1, n2))), abs(float(n1 @ n2))))


def check_planes(planes: Sequence[tuple[str, float, float]]) -> MassPlane:
    """
    Return the one plane of the mass of a galaxy that the planes of its
    components give, which must agree (to _PLANE_TOLERANCE, as the
    gravity takes them in one plane).

    Parameters
    ----------
    planes : Sequence of tuple
        The label (for the messages; e.g. "the disk 'gas'"), position
        angle and inclination (degrees) of each.

    Returns
    -------
    MassPlane
        The plane, that of the first.

    Raises
    ------
    ModelEvaluationError
        If there are none, or they disagree.
    """
    if not planes:
        raise ModelEvaluationError(
            "the mass of a galaxy lies in the plane of its components, "
            "but none of those that place it has a plane (a disk, or a "
            "spheroid of an axis)")
    first, posa, incl = planes[0]
    for label, other_posa, other_incl in planes[1:]:
        if plane_tilt((posa, incl), (other_posa, other_incl)) \
                > _PLANE_TOLERANCE:
            raise ModelEvaluationError(
                f"the mass of a galaxy lies in one plane, but the components "
                f"that place it disagree: {label} at posa {other_posa:g}, "
                f"incl {other_incl:g}, {first} at posa {posa:g}, incl "
                f"{incl:g}; tie them")
    return MassPlane(posa, incl)


@runtime_checkable
class SkyProjected(Protocol):
    """
    A mass component that can lay out its mass on the sky (see the
    module): every mass component but those that do not say where their
    mass is (tables of circular velocities, and functions of the user).
    """

    def takes_mass_plane(self) -> bool:
        """
        Check whether its layers lie in the plane of the mass of its
        galaxy (see SkyPlacement.plane), which the galaxy must have.
        """
        ...

    def sky_layers(
            self,
            params: ParamValues,
            context: 'MassContext',
            placement: SkyPlacement
    ) -> tuple[SkyLayer, ...]:
        """
        Return its layers on the sky.

        Parameters
        ----------
        params : ParamValues
            The values of its parameters, by name.
        context : MassContext
            Its context (see MassComponent.vcirc2), of a galaxy with a
            distance.
        placement : SkyPlacement
            Where its galaxy places its mass.

        Returns
        -------
        tuple of SkyLayer
            Its layers.

        Raises
        ------
        ModelEvaluationError
            If the values give no layers (e.g. a projected mass that is
            infinite, or a placement that disagrees).
        """
        ...


@runtime_checkable
class LinearBetweenKnots(Protocol):
    """
    A surface density that is linear in the radius between knots and 0
    beyond the last (e.g. one given at nodes): laid out on the sky as
    bands, exactly (see band_layer).
    """

    def linear_knots(self) -> np.ndarray:
        """
        Return the radii (kpc) between which it is linear: from its first,
        within which it is 0 (a hole) unless it is 0 (kpc), to its last,
        beyond which it is 0.
        """
        ...


def projected_ratio(q: npt.ArrayLike, incl: float) -> np.ndarray:
    """
    Return the axis ratio q' of the projection of a spheroid: q'^2 =
    cos^2(incl) + q^2 sin^2(incl), complete the square of m^2 along the
    line of sight (face-on 1, edge-on q).

    Parameters
    ----------
    q : array_like
        The axis ratio of the spheroid (its height over its width).
    incl : float
        The inclination of its axis (degrees).

    Returns
    -------
    np.ndarray
        The axis ratio of its projection, of the shape of q.
    """
    i = np.radians(incl)
    return np.sqrt(np.cos(i) ** 2 + np.asarray(q) ** 2 * np.sin(i) ** 2)


def plane_stretch(
        plane: MassPlane
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """
    Return how a sheet in a plane is laid out on the sky.

    Parameters
    ----------
    plane : MassPlane
        The plane.

    Returns
    -------
    tuple
        The position angle of its major axis (degrees), its stretch (1
        and |cos(incl)|) and the normal of the plane (see the module).
    """
    p, i = math.radians(plane.posa), math.radians(plane.incl)
    return (plane.posa, (1.0, abs(math.cos(i))),
            (math.sin(i) * math.cos(p), math.sin(i) * math.sin(p)))


def spheroid_stretch(
        q: float, posa: float, incl: float
) -> tuple[float, tuple[float, float]]:
    """
    Return how the projection of a spheroid is laid out on the sky:
    Sigma_sph(m') / q' on the ellipses m'^2 = x'^2 + y'^2 / q'^2 (x'
    along posa, its line of nodes; see projected_ratio), so a sheet of
    Sigma_sph stretched by 1 and q' along and across posa, or by q' and 1
    across and along it if q' > 1 (a prolate spheroid, longer along its
    projected axis).

    Parameters
    ----------
    q : float
        The axis ratio of the spheroid.
    posa, incl : float
        The position angle and the inclination of its axis (degrees).

    Returns
    -------
    tuple
        The position angle of the major axis of its projection (degrees)
        and its stretch (sigma1 >= sigma2).
    """
    ratio = float(projected_ratio(q, incl))
    if ratio > 1:
        return posa + 90, (ratio, 1.0)
    return posa, (1.0, ratio)


def height_slices(scale: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the slices of a thick layer: Gauss-Legendre nodes on each of
    the panels between the heights _HEIGHT_PANELS times its scale height,
    on both sides of its plane (for a pdf of the height p, the layer is
    the sum of its slices, each of the weight w p(h), see slice_weights).
    Their number is fixed, so that the layer changes smoothly with its
    parameters.

    Parameters
    ----------
    scale : float
        The scale height of the layer (kpc; e.g. see layer_scale).

    Returns
    -------
    tuple of np.ndarray
        The heights (kpc) of the slices and their widths (the weights of
        the quadrature).
    """
    nodes, weights = np.polynomial.legendre.leggauss(_HEIGHT_NODES)
    edges = scale * np.asarray(_HEIGHT_PANELS)
    middle, half = (edges[1:] + edges[:-1]) / 2, np.diff(edges) / 2
    heights = (middle[:, None] + half[:, None] * nodes).ravel()
    widths = (half[:, None] * weights).ravel()
    return (np.concatenate([-heights[::-1], heights]),
            np.concatenate([widths[::-1], widths]))


def layer_scale(z: np.ndarray, pdf: np.ndarray) -> float:
    """
    Return the scale height of a vertical distribution for its slices
    (see height_slices): its median height over ln 2 (the scale height of
    an exponential profile).

    Parameters
    ----------
    z : np.ndarray
        Heights (kpc), from 0, increasing, out to where it ends.
    pdf : np.ndarray
        Its pdf (even) at the heights.

    Returns
    -------
    float
        The scale height (kpc).
    """
    mass = np.concatenate(
        [[0], np.cumsum(np.diff(z) * (pdf[1:] + pdf[:-1]) / 2)])
    return float(np.interp(mass[-1] / 2, mass, z)) / math.log(2)


def check_slices(layer: 'EllipticalLayer | BandLayer', label: str) -> None:
    """
    Check that a thick layer (of slices) is not seen too close to edge-on
    to be laid out in slices: its median height times the tangent of its
    inclination (that of its band most edge-on, of those of mass) over its
    scale length (its half-mass radius over 1.678, the scale length of an
    exponential disk) at most _EDGE_ON_RATIO (see there).

    Parameters
    ----------
    layer : EllipticalLayer or BandLayer
        The layer (of a single slice, nothing to check).
    label : str
        What it is, for the message (e.g. "the mass disk 'gas'").

    Raises
    ------
    ModelEvaluationError
        If it is, saying the inclination it can be laid out to.
    """
    if len(layer.heights) < 2:
        return
    shape = _ellipse_shape if isinstance(layer, EllipticalLayer) \
        else _bands_shape
    slices, half, ratio = shape(layer)
    if not slices.sum() > 0:
        return
    order = np.argsort(np.abs(layer.heights))
    median = float(np.interp(
        slices.sum() / 2, np.cumsum(slices[order]),
        np.abs(layer.heights)[order]))
    scale = half / 1.678
    tangent = math.sqrt(max(1 - ratio ** 2, 0)) / max(ratio, 1e-300)
    if median * tangent > _EDGE_ON_RATIO * scale:
        limit = math.degrees(math.atan(_EDGE_ON_RATIO * scale / median))
        raise ModelEvaluationError(
            f"{label} is seen too close to edge-on (at "
            f"{math.degrees(math.atan(tangent)):.1f} degrees) to be laid "
            f"out on the sky in slices of its height: its median height "
            f"({median:.3g} kpc) times the tangent of its inclination is "
            f"{median * tangent / scale:.2g} times its scale length "
            f"({scale:.3g} kpc), above {_EDGE_ON_RATIO}, where its "
            f"deflection near its centre would be off by more than about "
            f"1e-3 of itself; it is laid out to {limit:.1f} degrees")


def _ellipse_shape(
        layer: EllipticalLayer
) -> tuple[np.ndarray, float, float]:
    """
    Return the mass of each slice of an elliptical layer (of a unit
    scale), its half-mass radius (kpc, in its plane) and its axis ratio.
    """
    table = layer.sigma
    log_radii = table.start + table.step * np.arange(table.values.shape[-1])
    # (the mass of each slice in each step of ln R, trapezoidal)
    steps = 2 * np.pi * np.exp(2 * log_radii) * table(np.exp(log_radii))
    steps = (steps[:, 1:] + steps[:, :-1]) / 2 * table.step
    cumulative = np.concatenate([[0], np.cumsum(steps.sum(axis=0))])
    half = math.exp(np.interp(cumulative[-1] / 2, cumulative, log_radii))
    return steps.sum(axis=1), half, layer.stretch[1] / layer.stretch[0]


def _bands_shape(layer: BandLayer) -> tuple[np.ndarray, float, float]:
    """
    Return the mass of each slice of bands (of a unit scale), their
    half-mass radius (kpc, in their planes) and the axis ratio of the
    band most edge-on of those that hold mass (1e-3 of it, at least).
    """
    inner, outer = layer.edges[:, 0], layer.edges[:, 1]
    powers = np.array([2, 3, 4])

    def antiderivative(R: np.ndarray) -> np.ndarray:
        # (of 2 pi R (c0 + c1 R + c2 R^2), by coefficient)
        return 2 * np.pi * R[:, None] ** powers / powers
    masses = np.sum(layer.density * (
        antiderivative(outer) - antiderivative(inner)), axis=-1)
    bands = masses.sum(axis=0)
    cumulative = np.concatenate([[0], np.cumsum(bands)])
    half = float(np.interp(
        cumulative[-1] / 2, cumulative, np.concatenate([inner[:1], outer])))
    held = bands >= 1e-3 * bands.sum()
    ratios = layer.stretch[held, 1] / layer.stretch[held, 0]
    return masses.sum(axis=1), half, float(ratios.min(initial=1.0))


def slice_weights(pdfs: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """
    Return the share of the surface density of each slice of a thick
    layer: the pdf of the height at each slice times its width (see
    height_slices), normalised to add up to 1 over the slices (so that the
    layer has the surface density of its pdf, whatever its tail beyond the
    slices).

    Parameters
    ----------
    pdfs : np.ndarray
        The pdf of the height at each slice: of shape (slices, ...), e.g.
        at radii.
    widths : np.ndarray
        The widths of the slices (see height_slices).

    Returns
    -------
    np.ndarray
        The shares, of the shape of pdfs; 0 where the pdfs are.
    """
    weighted = pdfs * widths.reshape((-1,) + (1,) * (pdfs.ndim - 1))
    total = weighted.sum(axis=0)
    return np.divide(
        weighted, total, out=np.zeros_like(weighted), where=total > 0)


def round_layer(
        centre: tuple[float, float],
        scale: float,
        enclosed_mass: Callable[[np.ndarray], np.ndarray]
) -> RoundLayer:
    """
    Return the round layer of a spherical mass: its projected mass within
    circles (see _numerics.projected_mass), on the radii of the tables.

    Parameters
    ----------
    centre : tuple of float
        Its centre (kpc, see the module).
    scale : float
        Its scale (see the module).
    enclosed_mass : Callable
        Its mass within spheres per unit scale (Msun), a function of the
        radii (kpc).

    Returns
    -------
    RoundLayer
        The layer.

    Raises
    ------
    ModelEvaluationError
        If its projected mass is infinite.
    """
    mass = _numerics.projected_mass(enclosed_mass, table_radii())
    return RoundLayer(
        _point(centre), float(scale),
        LogTable(_TABLE_START, _TABLE_STEP, _log(mass)))


def elliptical_layer(
        centre: tuple[float, float],
        scale: float,
        posa: float,
        stretch: tuple[float, float],
        sigma: Callable[[np.ndarray], np.ndarray],
        heights: np.ndarray | None = None,
        normal: tuple[float, float] = (0.0, 0.0)
) -> EllipticalLayer:
    """
    Return an elliptical layer (see EllipticalLayer), on the radii of the
    tables up to its tail (see _TAIL).

    Parameters
    ----------
    centre : tuple of float
        Its centre (kpc, see the module).
    scale : float
        Its scale (see the module).
    posa : float
        The position angle of its major axis (degrees).
    stretch : tuple of float
        sigma1 and sigma2 (see the module).
    sigma : Callable
        The surface density of each slice per unit scale (Msun/kpc^2, in
        its plane), a function of the radii R (kpc) that returns an array
        of shape (slices, len(R)).
    heights : np.ndarray, optional
        The heights of the slices (kpc); one slice in its plane without.
    normal : tuple of float, optional
        The normal of its plane (see the module); none without heights.

    Returns
    -------
    EllipticalLayer
        The layer.
    """
    radii = table_radii()
    values = np.atleast_2d(sigma(radii))
    mass = np.abs(values).max(axis=0) * radii ** 2
    keep = max(4, int(np.flatnonzero(mass >= _TAIL * mass.max())[-1]) + 1) \
        if mass.max() > 0 else 4
    return EllipticalLayer(
        _point(centre), float(scale), float(posa),
        (float(stretch[0]), float(stretch[1])),
        LogTable(_TABLE_START, _TABLE_STEP, _log(values[:, :keep])),
        np.zeros(1) if heights is None else np.asarray(heights, dtype=float),
        _point(normal))


def band_layer(
        scale: float,
        edges: np.ndarray,
        geometry: Callable[[np.ndarray], tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        sigma: Callable[[np.ndarray], np.ndarray],
        heights: np.ndarray | None = None
) -> BandLayer:
    """
    Return the bands between consecutive radii (kpc, in their plane) of a
    surface density of each slice per unit scale (Msun/kpc^2; a function
    of the radii R, kpc, which returns an array of shape (slices,
    len(R))) that is a quadratic in R within each band: its values at the
    edges and the middle of each give its coefficients (exactly, for a
    density linear, or a product of two linear functions, there).

    Parameters
    ----------
    scale : float
        Its scale (see the module).
    edges : np.ndarray
        The radii between the bands (kpc), increasing.
    geometry : Callable
        Of the radii of the middles of the bands, their centres (kpc,
        (bands, 2)), the position angles of their major axes (degrees),
        their stretch ((bands, 2)) and the normals of their planes
        ((bands, 2)).
    sigma : Callable
        The surface density of each slice (see above).
    heights : np.ndarray, optional
        The heights of the slices (kpc); one slice in its plane without.

    Returns
    -------
    BandLayer
        The bands.
    """
    edges = np.asarray(edges, dtype=float)
    inner, outer = edges[:-1], edges[1:]
    middle = (inner + outer) / 2
    centres, posa, stretch, normals = geometry(middle)
    at = np.atleast_2d(sigma(np.concatenate([inner, middle, outer])))
    s0, s1, s2 = np.split(at, 3, axis=-1)
    # (the quadratic through the values at R_in, R_mid and R_out)
    width = outer - inner
    c2 = 2 * (s0 - 2 * s1 + s2) / width ** 2
    slope = (s2 - s0) / width - c2 * (inner + outer)
    c0 = s1 - slope * middle - c2 * middle ** 2
    return BandLayer(
        float(scale), np.asarray(centres, dtype=float),
        np.asarray(posa, dtype=float), np.asarray(stretch, dtype=float),
        np.stack([inner, outer], axis=1), np.stack([c0, slope, c2], axis=-1),
        np.zeros(1) if heights is None else np.asarray(heights, dtype=float),
        np.asarray(normals, dtype=float))


def fixed_geometry(
        centre: tuple[float, float],
        posa: float,
        stretch: tuple[float, float],
        normal: tuple[float, float] = (0.0, 0.0)
) -> Callable[[np.ndarray], tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return the geometry of bands of one centre, position angle, stretch
    and normal.

    Parameters
    ----------
    centre : tuple of float
        The centre (kpc, see the module).
    posa : float
        The position angle of their major axes (degrees).
    stretch : tuple of float
        sigma1 and sigma2 (see the module).
    normal : tuple of float, optional
        The normal of their plane; none by default (of thin bands).

    Returns
    -------
    Callable
        Their geometry (see band_layer).
    """
    def geometry(radii: np.ndarray) -> tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(radii)
        return (np.tile(centre, (n, 1)), np.full(n, float(posa)),
                np.tile(stretch, (n, 1)), np.tile(normal, (n, 1)))
    return geometry


def ring_geometry(
        placement: SkyPlacement, component: str
) -> Callable[[np.ndarray], tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return the geometry of bands on the rings of the disk of a component:
    of each, that of its ring at the middle of the band (a tilted ring of
    the mass, as the disk is drawn between its rings to second order in
    the change of its geometry across a band): its centre, lifted along
    the normal of its plane, and the image of its plane on the sky, tilted
    by its tilt, stretched along its singular vectors (see RingGeometry).

    Parameters
    ----------
    placement : SkyPlacement
        Where the galaxy places its mass.
    component : str
        The name of the component.

    Returns
    -------
    Callable
        The geometry (see band_layer).
    """
    def geometry(radii: np.ndarray) -> tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rings = placement.rings(component, radii)
        p, i = np.radians(rings.posa), np.radians(rings.incl)
        sin_p, cos_p, sin_i, cos_i = np.sin(p), np.cos(p), np.sin(i), np.cos(i)
        normals = np.stack([sin_i * cos_p, sin_i * sin_p], axis=1)
        # The image on the sky of the plane of each ring (x_d, y_d): that
        # of transform_cpos_posa_incl_inverse, plus its tilt along normal
        image = np.empty((len(radii), 2, 2))
        image[:, 0, 0] = -sin_p + normals[:, 0] * rings.tilt_x
        image[:, 0, 1] = -cos_i * cos_p + normals[:, 0] * rings.tilt_y
        image[:, 1, 0] = cos_p + normals[:, 1] * rings.tilt_x
        image[:, 1, 1] = -cos_i * sin_p + normals[:, 1] * rings.tilt_y
        vectors, stretch, _ = np.linalg.svd(image)
        major = vectors[:, :, 0]
        posa = np.degrees(np.arctan2(-major[:, 0], major[:, 1]))
        centres = np.stack([rings.x, rings.y], axis=1) \
            + rings.lift[:, None] * normals
        return centres, posa, stretch, normals
    return geometry


def _point(values: npt.ArrayLike) -> tuple[float, float]:
    """Return a point (two floats) of two values."""
    x, y = np.asarray(values, dtype=float)
    return float(x), float(y)
