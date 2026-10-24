"""
The light of the image traits of the disks of a galaxy (ImageMaps): the
face-on light of each, which the galaxy makes from its image at the
geometry of its disk at each evaluation (see traits.RPTraitImage).
"""

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from gbkfit.params import ParamValues
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from ._detail import disk_to_sky
from .disks._disk import TRAIT_KINDS
from .traits import RPTraitImage, map_cell_centres

if TYPE_CHECKING:
    from ._component_set import ComponentSet
    from .disks._disk import Disk


__all__ = [
    'ImageMaps'
]


# The inclination (degrees) above which the image says little of the
# face-on light (see ImageMaps.warn_inclination)
_EDGE_ON_WARNING = 80.0

# The parameters of the geometry of a disk, by their names in it
_GEOMETRY = ('xpos', 'ypos', 'posa', 'incl')


class _ImageTrait(NamedTuple):
    """
    An image trait of a disk: the trait, the names of its parameters in
    the disk, by their names in the trait, and the centres of its cells
    in the plane of the disk (x, y; arcsec) and their radii.
    """
    trait: RPTraitImage
    names: dict[str, str]
    x: np.ndarray
    y: np.ndarray
    radii: np.ndarray


class ImageMaps:
    """
    The face-on light of the image traits of the disks of a galaxy (see
    traits.RPTraitImage), from the values of their parameters, in the
    frame of the parameters of the galaxy, whose origin on the sky is at
    the given RA and Dec: that of the first observation on the sky (see
    ObservationGroup). A warped disk takes the image at the geometry of
    each cell's radius, its geometry interpolated (with the interpolation
    of the disk) between its first node and its last. The galaxy makes them
    at each evaluation, for all that reads the values of its disks (see
    GravityPlan).

    Parameters
    ----------
    component_set : ComponentSet
        The components of the galaxy.
    sky_origin : tuple of float, optional
        The RA and Dec (degrees) of the origin of the frame; required when
        a disk has an image trait.

    Raises
    ------
    ConfigError
        If a disk has an image trait, but there is no sky origin.
    """

    def __init__(
            self,
            component_set: 'ComponentSet',
            sky_origin: tuple[float, float] | None
    ):
        self._component_set = component_set
        self._sky_origin = sky_origin
        self._traits: dict[int, list[_ImageTrait]] = {}
        for i, component in enumerate(component_set.components()):
            disk = component.density_disk()
            if disk is None or not disk.image_params():
                continue
            for kind in TRAIT_KINDS:
                for trait, names in zip(
                        disk.traits(kind), disk.trait_params(kind).pnames):
                    if isinstance(trait, RPTraitImage):
                        x, y = map_cell_centres(trait.size(), trait.step())
                        self._traits.setdefault(i, []).append(_ImageTrait(
                            trait, names, x, y, np.hypot(x, y)))
        if self._traits and sky_origin is None:
            raise ConfigError(
                "the light of an image trait is placed on the sky by the "
                "world coordinates of its image, which need the origin of "
                "the frame of the model: an observation on the sky")
        components = component_set.components()
        for i, images in self._traits.items():
            label = components[i].name() or f'component{i}'
            for image in images:
                image.trait.check_coverage(
                    image.x, image.y, sky_origin, f"the disk of {label!r}")

    def indices(self) -> tuple[int, ...]:
        """Return the indices of the components with image traits."""
        return tuple(self._traits)

    def values(self, index: int, params: ParamValues) -> ParamValues:
        """
        Return the face-on light of the image traits of the disk of the
        component of the given index, from the values of its parameters
        (by its own names), by the names of their parameters in it (see
        traits.Trait.image_params).
        """
        disk = self._component_set.components()[index].density_disk()
        values = {}
        for image in self._traits[index]:
            geometry = _geometry(disk, params, image.radii)
            x, y, _ = disk_to_sky(
                image.x, image.y, 0, *(geometry[p] for p in _GEOMETRY))
            own = {name: params[full] for name, full in image.names.items()
                   if name not in image.trait.image_params()}
            values[image.names['m']] = image.trait.face_on(
                own, x, y, geometry['incl'], self._sky_origin)
        return values

    def warn_inclination(self, params: ParamValues) -> None:
        """
        Warn (see parseutils.warn) of the disks with image traits whose
        inclination (by the values of the parameters of the galaxy) is
        above _EDGE_ON_WARNING: near edge-on, the image says little of the
        face-on light, which the deprojection stretches by 1 / cos(incl)
        along the minor axis.
        """
        components = self._component_set.components()
        mappings = self._component_set.mappings()
        for i in self._traits:
            incl = np.max(params[mappings[i]['incl']])
            if incl > _EDGE_ON_WARNING:
                label = components[i].name() or f'component{i}'
                parseutils.warn(
                    f"the disk of {label!r} is at the inclination "
                    f"{incl:.3g} degrees: its image traits stretch their "
                    f"images by 1 / cos(incl) = "
                    f"{1 / np.cos(np.radians(incl)):.3g} along its minor "
                    f"axis, which say little of its face-on light")


def _geometry(
        disk: 'Disk', params: ParamValues, radii: np.ndarray
) -> dict[str, np.ndarray | float]:
    """
    Return the geometry of a disk (see _GEOMETRY) at the given radii (in
    its plane, arcsec), from the values of its parameters: their values, or
    those at its nodes interpolated with its interpolation, at the radii
    within its first node and its last.
    """
    rnodes = np.asarray(disk.rnodes())
    clipped = np.clip(radii, rnodes[0], rnodes[-1])
    geometry = {}
    for name in _GEOMETRY:
        value = np.asarray(params[name], dtype=float)
        geometry[name] = value if value.ndim == 0 \
            else disk.interp()(rnodes, value)(clipped)
    return geometry
