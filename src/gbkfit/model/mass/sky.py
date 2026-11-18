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
inclination i is stretched by 1 and cos(i). A thick layer is the
integral over the height h of such sheets, each moved by h along the
normal of its plane (a vector of the sky, of length sin(i)), of the
surface density Sigma(R) times the shape of its vertical structure at R
and h (see HeightProfile): lensing integrates it at each point, to any
inclination.
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
    'HeightProfile',
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

# The values of the tables of the pdfs of the height of the terms of a
# thick layer (see HeightProfile), evenly from 0 to their tops: cubic
# between them, to about 1e-7 of an exponential pdf of 24 scale heights
_HEIGHT_TABLE = 512

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
    field equal to the last bits but scale (and those of their tables and
    vertical structures).
    """
    if type(first) is not type(second):
        return False
    for field in dataclasses.fields(first):
        if field.name == 'scale':
            continue
        a, b = getattr(first, field.name), getattr(second, field.name)
        if dataclasses.is_dataclass(a) or dataclasses.is_dataclass(b):
            if not _same(a, b):
                return False
        elif a is None or b is None:
            if a is not b:
                return False
        elif not np.array_equal(a, b):
            return False
    return True


@dataclasses.dataclass(frozen=True, eq=False)
class HeightProfile:
    """
    The vertical structure of a thick layer: the sum of its terms (e.g.
    those of the density of a disk component), each a pdf of the height
    times its share, at its knots (radii in its plane), and linear in the
    radius between them (held beyond the first and the last knots): the
    shape of the layer, the density of its sheet at the height h per unit
    of its surface density Sigma(R) (see the module). One knot of a shape
    that is the same at every radius (e.g. a law).

    Attributes
    ----------
    knots : np.ndarray
        The radii of the knots (kpc, in its plane), increasing.
    shares : np.ndarray
        The share of each term at each knot: of shape (terms, knots).
    pdfs : np.ndarray
        The pdf of the height (1/kpc, even in the height) of each term at
        each knot, at _HEIGHT_TABLE heights evenly from 0 to its top (cubic
        between them, see gbkfit/lensing/layers.hpp): of shape (terms,
        knots, _HEIGHT_TABLE).
    tops : np.ndarray
        The height (kpc) of each term above which its pdf is 0: where it
        ends (e.g. the zmax of a density height trait), or where its tails
        hold no mass that matters (e.g. of a law).
    scales : np.ndarray
        The scale height (kpc) of each term (see layer_scale), the
        smallest of its knots.
    """
    knots: np.ndarray
    shares: np.ndarray
    pdfs: np.ndarray
    tops: np.ndarray
    scales: np.ndarray


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
    its plane, stretched onto the sky (see the module), scale times its
    table (Msun/kpc^2 per unit scale, in its plane); thin, or thick, of
    its vertical structure, its sheets moved by their heights along the
    normal of its plane. The surface density of the sky of a thin layer is
    Sigma(a / sigma1) / (sigma1 sigma2) on the ellipse of the semi-major
    axis a (along posa) and the axis ratio sigma2 / sigma1, and 0 beyond
    the last radius of its table.

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
        Its surface density per unit scale (Msun/kpc^2, in its plane).
    heights : HeightProfile or None
        Its vertical structure; None, thin.
    normal : tuple of float
        The vector of the sky (kpc per kpc of height) of the normal of its
        plane.
    """
    centre: tuple[float, float]
    scale: float
    posa: float
    stretch: tuple[float, float]
    sigma: LogTable
    heights: HeightProfile | None
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
    draws it (homoeoid bands): scale times the quadratic of each band
    (Msun/kpc^2 per unit scale, in its plane). The bands add up,
    overlapping or not. Thick, of their vertical structure, each its
    sheets moved by their heights along its normal: within a band, the
    shapes at the knots below and above it times the quadratics of the two
    rows of its coefficients (exact where the surface density is linear
    within the band: its knots are edges of bands, see band_layer).

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
        The coefficients c0, c1, c2 of the quadratics c0 + c1 R + c2 R^2 of
        each band (of its surface density, in its first row; thick, of the
        shapes at its two knots): of shape (bands, 2, 3).
    knots : np.ndarray
        The indices of the knots below and above each band of the knots of
        its vertical structure (int): of shape (bands, 2); zeros, thin.
    heights : HeightProfile or None
        Their vertical structure; None, thin.
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
    knots: np.ndarray
    heights: HeightProfile | None
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


def layer_scale(z: np.ndarray, pdf: np.ndarray) -> float:
    """
    Return the scale height of a vertical distribution: its median height
    over ln 2 (the scale height of an exponential profile).

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


def height_profile(
        knots: npt.ArrayLike,
        shares: npt.ArrayLike,
        tops: Sequence[float],
        pdfs: Callable[[np.ndarray], np.ndarray]
) -> HeightProfile:
    """
    Return the vertical structure of a thick layer (see HeightProfile).

    Parameters
    ----------
    knots : array_like
        The radii of its knots (kpc, in its plane), increasing.
    shares : array_like
        The share of each term at each knot: of shape (terms, knots).
    tops : Sequence of float
        The height (kpc) of each term above which its pdf is 0.
    pdfs : Callable
        The pdf of the height (1/kpc) of every term at every knot, a
        function of the heights z (kpc, from 0) that returns an array of
        shape (terms, knots, len(z)).

    Returns
    -------
    HeightProfile
        The vertical structure.
    """
    tops = np.asarray(tops, dtype=float)
    tables, scales = [], []
    for term, top in enumerate(tops):
        z = np.linspace(0, top, _HEIGHT_TABLE)
        table = np.asarray(pdfs(z), dtype=float)[term]
        tables.append(table)
        scales.append(min(layer_scale(z, row) for row in table))
    return HeightProfile(
        np.asarray(knots, dtype=float), np.asarray(shares, dtype=float),
        np.stack(tables), tops, np.array(scales))


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
        heights: HeightProfile | None = None,
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
        Its surface density per unit scale (Msun/kpc^2, in its plane), a
        function of the radii R (kpc).
    heights : HeightProfile, optional
        Its vertical structure; thin without.
    normal : tuple of float, optional
        The normal of its plane (see the module); none thin.

    Returns
    -------
    EllipticalLayer
        The layer.
    """
    radii = table_radii()
    values = np.asarray(sigma(radii), dtype=float)
    mass = np.abs(values) * radii ** 2
    keep = max(4, int(np.flatnonzero(mass >= _TAIL * mass.max())[-1]) + 1) \
        if mass.max() > 0 else 4
    return EllipticalLayer(
        _point(centre), float(scale), float(posa),
        (float(stretch[0]), float(stretch[1])),
        LogTable(_TABLE_START, _TABLE_STEP, _log(values[:keep])), heights,
        _point(normal))


def band_layer(
        scale: float,
        edges: np.ndarray,
        geometry: Callable[[np.ndarray], tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        sigma: Callable[[np.ndarray], np.ndarray],
        heights: HeightProfile | None = None
) -> BandLayer:
    """
    Return the bands between consecutive radii (kpc, in their plane) and
    the knots of their vertical structure within them (so that the shape
    is linear within each band), of a surface density per unit scale
    (Msun/kpc^2; a function of the radii R, kpc) that is a quadratic in R
    within each band: its values at the edges and the middle of each give
    its coefficients (exactly, for a density linear there, also times the
    shape, linear between the knots).

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
        The surface density (see above).
    heights : HeightProfile, optional
        Their vertical structure; thin without.

    Returns
    -------
    BandLayer
        The bands.
    """
    edges = np.asarray(edges, dtype=float)
    if heights is not None:
        knots = heights.knots
        edges = np.union1d(
            edges, knots[(knots > edges[0]) & (knots < edges[-1])])
    inner, outer = edges[:-1], edges[1:]
    middle = (inner + outer) / 2
    centres, posa, stretch, normals = geometry(middle)
    radii = np.concatenate([inner, middle, outer])
    at = np.asarray(sigma(radii), dtype=float)
    below = np.zeros(len(middle), dtype=np.int32)
    if heights is None or len(heights.knots) == 1:
        parts = (at, np.zeros_like(at))
    else:
        # (the shapes at the knots around each band, linear between them)
        knots = heights.knots
        below = np.clip(np.searchsorted(knots, middle, side='right') - 1,
                        0, len(knots) - 2).astype(np.int32)
        k = np.tile(below, 3)
        t = np.clip((radii - knots[k]) / (knots[k + 1] - knots[k]), 0, 1)
        parts = (at * (1 - t), at * t)
    # (the quadratic through the values at R_in, R_mid and R_out)
    width = outer - inner
    coefficients = []
    for values in parts:
        s0, s1, s2 = np.split(values, 3)
        c2 = 2 * (s0 - 2 * s1 + s2) / width ** 2
        slope = (s2 - s0) / width - c2 * (inner + outer)
        c0 = s1 - slope * middle - c2 * middle ** 2
        coefficients.append(np.stack([c0, slope, c2], axis=-1))
    above = below + (heights is not None and len(heights.knots) > 1)
    return BandLayer(
        float(scale), np.asarray(centres, dtype=float),
        np.asarray(posa, dtype=float), np.asarray(stretch, dtype=float),
        np.stack([inner, outer], axis=1), np.stack(coefficients, axis=1),
        np.stack([below, above], axis=1).astype(np.int32), heights,
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
