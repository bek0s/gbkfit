"""
The frames of the observations of a group (see ObservationGroup): where
the origin of the frame of the first observation on the sky is in the
frame of each (see alignments), how each moves what it sees into its
frame (see FrameShift), and the checks of their offsets and frames of
rest.
"""

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from astropy.coordinates import SkyCoord

from gbkfit.config import warn
from gbkfit.model import Galaxy
from gbkfit.params import ParamValues
from .observation import SkyFrame


__all__ = [
    'FrameShift',
    'alignments',
    'check_offsets',
    'first_on_sky',
    'frame_shift',
    'warn_frames'
]


# Warn of observations whose origins are further apart on the sky (deg),
# or whose frames turn by more (deg; see alignments)
_ALIGNMENT_SEPARATION_WARNING = 1.0
_ALIGNMENT_ROTATION_WARNING = 0.1


class FrameShift(NamedTuple):
    """
    How an observation moves what it sees into its frame (see
    ObservationGroup): the parameters of the centres (x and y) and of the
    systemic velocities (v) of the components of the galaxies it sees,
    where the origin of the frame of the first observation is in its
    frame (arcsec, x and y), and the names of the parameters of the
    offsets of its frame, of x, y and the velocity (None for those it
    does not fit).
    """
    x: tuple[str, ...]
    y: tuple[str, ...]
    v: tuple[str, ...]
    alignment: tuple[float, float]
    offsets: tuple[str | None, str | None, str | None]

    def shifts(self, params: ParamValues) -> tuple[float, float, float]:
        """
        Return the shifts of the centres (x and y) and of the systemic
        velocities, by the alignment and the offsets.
        """
        offsets = [
            0.0 if name is None else params[name] for name in self.offsets]
        return (
            self.alignment[0] + offsets[0], self.alignment[1] + offsets[1],
            offsets[2])

    def moved(self, params: ParamValues) -> ParamValues:
        """
        Return the parameters with the centres and the systemic velocities
        moved by the alignment and the offsets (the parameters themselves
        without either).
        """
        if self.alignment == (0, 0) and self.offsets == (None, None, None):
            return params
        moved = dict(params)
        for names, shift in zip((self.x, self.y, self.v), self.shifts(params)):
            for name in names:
                moved[name] = params[name] + shift
        return moved


def frame_shift(
    galaxies: Sequence[Galaxy],
    mappings: Sequence[Mapping[str, str]],
    seen: Sequence[int],
    frame: SkyFrame | None,
    alignment: tuple[float, float]
) -> FrameShift:
    """
    Return how an observation that sees the galaxies of the given
    indices, of the galaxies of a group and the mappings of their
    parameters to those of the group, of the given frame on the sky (None
    if it is not on the sky), moves them into its frame, with the given
    alignment (see FrameShift).
    """
    names = dict(x=[], y=[], v=[])
    for g in seen:
        galaxy, mapping = galaxies[g], mappings[g]
        for i in range(len(galaxy.components())):
            placed = galaxy.component_frame_params(i, mapping)
            for key, name in zip(
                    'xyv', (placed.x, placed.y, placed.vsys)):
                if name is not None and name not in names[key]:
                    names[key].append(name)
    return FrameShift(
        tuple(names['x']), tuple(names['y']), tuple(names['v']),
        alignment,
        (None, None, None) if frame is None else frame.offsets)


def first_on_sky(frames: Sequence[SkyFrame | None]) -> int | None:
    """
    Return the index of the first observation on the sky, of those of
    the given frames on the sky (None for those that are not on it; see
    Observation.sky_frame), whose frame is that of the parameters of the
    galaxies; None if none is.
    """
    return next(
        (i for i, frame in enumerate(frames) if frame is not None), None)


def alignments(
        frames: Sequence[SkyFrame | None]
) -> list[tuple[float, float]]:
    """
    Return where the origin of the frame of the first observation on the
    sky is in the frame of each, of those of the given frames on the sky
    (arcsec, x to the west and y to the north), of the RA and Dec of their
    origins (see Observation.sky_frame); (0, 0) for those that are not on
    the sky (None).
    Warn (see warn) of origins more than
    _ALIGNMENT_SEPARATION_WARNING apart (e.g. one of data without
    coordinates, at RA and Dec 0), and of frames that turn by more than
    _ALIGNMENT_ROTATION_WARNING: the frames are aligned by a shift; that
    of an origin far from the first, near a pole, also turns by about the
    separation times tan(Dec).
    """
    f = first_on_sky(frames)
    if f is None:
        return [(0.0, 0.0)] * len(frames)
    origins = [None if frame is None else frame.origin for frame in frames]
    first = SkyCoord(*origins[f], unit='deg')
    alignments = []
    for i, origin in enumerate(origins):
        if origin is None:
            alignments.append((0.0, 0.0))
            continue
        origin = SkyCoord(*origin, unit='deg')
        east, north = origin.spherical_offsets_to(first)
        alignments.append(
            (-float(east.to_value('arcsec')),
             float(north.to_value('arcsec'))))
        separation = origin.separation(first).to_value('deg')
        if separation > _ALIGNMENT_SEPARATION_WARNING:
            warn(
                f"the origins of the frames of observations {f} and {i} (the "
                f"RA and Dec of their reference pixels) are {separation:g} "
                f"deg apart: check their coordinates")
        elif separation > 0:
            # (the convergence of the meridians: the bearing from the origin
            # to the first, less the bearing back, less 180 deg)
            bearings = origin.position_angle(first) \
                - first.position_angle(origin)
            rotation = bearings.to_value('deg') % 360 - 180
            if abs(rotation) > _ALIGNMENT_ROTATION_WARNING:
                warn(
                    f"the frames of observations {f} and {i} turn by "
                    f"{rotation:.2g} deg against each other, which their "
                    f"alignment leaves out: their reference pixels are "
                    f"{separation * 60:.2g} arcmin apart")
    return alignments


def check_offsets(
        galaxies: Sequence[Galaxy],
        frames: Sequence[SkyFrame | None],
        seen: Sequence[Sequence[int]]
) -> None:
    """
    Raise ValueError if every observation on the sky that sees a galaxy
    has offsets of position, or of velocity (see ObservationSky), of the
    frame on the sky of each observation (None for those that are not on
    it) and the galaxies each sees: they would move its centre, or its
    systemic velocity, with nothing to place it.
    """
    for g, galaxy in enumerate(galaxies):
        offsets = [
            frame.offsets
            for frame, indices in zip(frames, seen)
            if g in indices and frame is not None]
        for kind, k in (('position', 0), ('velocity', 2)):
            if offsets and all(params[k] is not None for params in offsets):
                raise ValueError(
                    f"every observation of galaxy {galaxy.name() or g} has "
                    f"offsets of {kind}; one needs none, whose frame places "
                    f"the galaxy")


def warn_frames(
        galaxies: Sequence[Galaxy],
        frames: Sequence[SkyFrame | None],
        seen: Sequence[Sequence[int]]
) -> None:
    """
    Warn (see warn) if the observations that see a galaxy
    without offsets of velocity have velocities in different frames of
    rest (see SkyFrame), of the frame on the sky of each observation
    (None for those that are not on it) and the galaxies each sees: their
    systemic velocities differ (by up to some 30 km/s).
    """
    for g, galaxy in enumerate(galaxies):
        rests = {
            i: frame.velocity_frame
            for i, (frame, indices) in enumerate(zip(frames, seen))
            if g in indices and frame is not None
            and frame.velocity_frame is not None
            and frame.offsets[2] is None}
        if len(set(rests.values())) > 1:
            warn(
                f"the observations of galaxy {galaxy.name() or g} have "
                f"velocities in different frames of rest (by observation: "
                f"{rests}), whose systemic velocities differ by up to some "
                f"30 km/s: give all but one offsets of velocity (offsets: "
                f"{{velocity: true}}), or data in one frame")
