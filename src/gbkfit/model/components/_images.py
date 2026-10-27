"""
The light of the image traits of a disk (ImageMaps): the face-on light of
each, which the stage of its galaxy makes from its image at the geometry
of the disk at each evaluation (see traits.RPTraitImage).
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
    from .disks._disk import NodalDisk


__all__ = [
    'ImageMaps'
]


# The inclination (degrees) above which the image says little of the
# face-on light (see ImageMaps.warn_inclination)
_EDGE_ON_WARNING = 80.0

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
    The face-on light of the image traits of a disk (see
    traits.RPTraitImage), from the values of its parameters, in the frame
    of the parameters of its galaxy, whose origin on the sky is at the
    given RA and Dec: that of the first observation on the sky (see
    ObservationGroup). A warped disk takes the image at the geometry of
    each cell's radius, its geometry interpolated (with the interpolation
    of the disk) between its first node and its last. The stage of the
    galaxy makes them at each evaluation, for all that reads the values of
    the disk (see DiskStage).

    Parameters
    ----------
    disk : NodalDisk
        The disk, with image traits.
    label : str
        The name of its component, or its position, for the messages.
    sky_origin : tuple of float, optional
        The RA and Dec (degrees) of the origin of the frame; required.

    Raises
    ------
    ConfigError
        If there is no sky origin, or the image of a trait does not cover
        the disk (see RPTraitImage.check_coverage).
    """

    def __init__(
            self,
            disk: 'NodalDisk',
            label: str,
            sky_origin: tuple[float, float] | None
    ):
        self._disk = disk
        self._label = label
        self._sky_origin = sky_origin
        self._traits: list[_ImageTrait] = []
        for kind in TRAIT_KINDS:
            for trait, names in zip(
                    disk.traits(kind), disk.trait_params(kind).pnames):
                if isinstance(trait, RPTraitImage):
                    x, y = map_cell_centres(trait.size(), trait.step())
                    self._traits.append(_ImageTrait(
                        trait, names, x, y, np.hypot(x, y)))
        if sky_origin is None:
            raise ConfigError(
                "the light of an image trait is placed on the sky by the "
                "world coordinates of its image, which need the origin of "
                "the frame of the model: an observation on the sky")
        for image in self._traits:
            image.trait.check_coverage(
                image.x, image.y, sky_origin, f"the disk of {label!r}")

    def values(self, params: ParamValues) -> ParamValues:
        """
        Return the face-on light of its image traits, from the values of
        the parameters of the disk (by its own names), by the names of
        their parameters in it (see traits.ImageInput).
        """
        values = {}
        for image in self._traits:
            xpos, ypos, posa, incl = _geometry(
                self._disk, params, image.radii)
            x, y, _ = disk_to_sky(image.x, image.y, 0, xpos, ypos, posa, incl)
            own = {name: params[full] for name, full in image.names.items()
                   if name not in image.trait.from_galaxy().params}
            values[image.names['m']] = image.trait.face_on(
                own, x, y, incl, self._sky_origin)
        return values

    def warn_inclination(self, params: ParamValues) -> None:
        """
        Warn (see parseutils.warn) if the inclination of the disk (by the
        values of its parameters) is above _EDGE_ON_WARNING: near edge-on,
        the image says little of the face-on light, which the deprojection
        stretches by 1 / cos(incl) along the minor axis.
        """
        # (the nearest to edge-on, from either side)
        incls = np.atleast_1d(params[self._disk.frame_params().incl])
        cosines = np.abs(np.cos(np.radians(incls)))
        if np.min(cosines) < np.cos(np.radians(_EDGE_ON_WARNING)):
            incl, cosine = incls[np.argmin(cosines)], np.min(cosines)
            parseutils.warn(
                f"the disk of {self._label!r} is at the inclination "
                f"{incl:.3g} degrees: its image traits stretch their "
                f"images by 1 / |cos(incl)| = {1 / cosine:.3g} along its "
                f"minor axis, which say little of its face-on light")


def _geometry(
        disk: 'NodalDisk', params: ParamValues, radii: np.ndarray
) -> tuple[np.ndarray | float, ...]:
    """
    Return the centre (xpos, ypos), the position angle and the inclination
    of a disk at the given radii (in its plane, arcsec), from the values of
    its parameters: their values, or those at its nodes interpolated with
    its interpolation, at the radii within its first node and its last.
    """
    frame = disk.frame_params()
    rnodes = np.asarray(disk.rnodes())
    clipped = np.clip(radii, rnodes[0], rnodes[-1])
    geometry = []
    for name in (frame.x, frame.y, frame.posa, frame.incl):
        value = np.asarray(params[name], dtype=float)
        geometry.append(value if value.ndim == 0
                        else disk.interp()(rnodes, value)(clipped))
    return tuple(geometry)
