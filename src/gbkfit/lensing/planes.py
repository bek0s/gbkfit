"""
The planes of the light of a lensed sky (see plane_layout): galaxies
and lenses ordered by redshift, those close in redshift sharing a plane
at one distance (see Plane), whose light the lenses of the planes in
front of it bend (see Tracer).
"""

from collections.abc import Sequence
from typing import NamedTuple

import astropy.constants


__all__ = [
    'PLANE_LINKING_VELOCITY',
    'Plane',
    'plane_layout',
    'velocity_separation'
]


# The speed of light (km/s)
_C = astropy.constants.c.to_value('km/s')

# The velocity separation (km/s) within which galaxies and lenses share a
# plane: within a group or a cluster, their differences of redshift are
# their peculiar velocities (up to some 3 times a dispersion of 1000
# km/s), not their distances
PLANE_LINKING_VELOCITY = 3000.0


class Plane(NamedTuple):
    """
    A plane of the light of a lensed sky (see plane_layout): galaxies and
    lenses at one distance, by their indices (in the order they were
    given), and the redshift of that distance.

    The lenses of a plane do not bend the light of its galaxies; they
    bend that of the planes behind it, their deflections added (see
    Tracer).
    """
    redshift: float
    galaxies: tuple[int, ...]
    lenses: tuple[int, ...]


def velocity_separation(z1: float, z2: float) -> float:
    """
    Return the velocity separation of two redshifts: c (z2 - z1) /
    (1 + z1).

    Parameters
    ----------
    z1, z2 : float
        The redshifts.

    Returns
    -------
    float
        The separation (km/s), positive if z2 is the further.
    """
    return _C * (z2 - z1) / (1 + z1)


def plane_layout(
        galaxies: Sequence[tuple[str | None, float | None]],
        lenses: Sequence[tuple[str, float]],
        depth_order: Sequence[Sequence[str]] = ()
) -> tuple[Plane, ...]:
    """
    Return the planes of galaxies and lenses, near to far.

    The galaxies and the lenses are sorted by redshift, and neighbours
    within PLANE_LINKING_VELOCITY of each other share a plane (friends of
    friends: a plane can span more, through members between). The
    galaxies of a group of the depth order (that pass through each other)
    must share one: the redshifts place them.

    The distance of a plane is the mean redshift of its lenses, or of its
    galaxies if it has no lens: that of its lenses, which their strengths
    are quoted at (see Lens), and one for all, so that the light of the
    plane meets them all at once.

    Parameters
    ----------
    galaxies : Sequence of tuple
        The name (None for a galaxy without one) and the redshift of each
        galaxy.
    lenses : Sequence of tuple
        The name and the redshift of each lens.
    depth_order : Sequence of Sequence of str, optional
        The order of galaxies along the line of sight, front to back (see
        ObservationGroup): the names of the galaxies of each of its items,
        one for a galaxy alone or several for a group of them.

    Returns
    -------
    tuple of Plane
        The planes, near to far.

    Raises
    ------
    ValueError
        If there are no lenses, a galaxy has no redshift, the depth order
        names unknown galaxies or repeats them, has a group of galaxies of
        several planes, or puts a galaxy in front of one that a plane
        nearer than its own holds.
    """
    if not lenses:
        raise ValueError("planes are those of lenses; there are none")
    if missing := [
            name if name is not None else i
            for i, (name, redshift) in enumerate(galaxies)
            if redshift is None]:
        raise ValueError(
            f"with lenses, every galaxy needs a redshift, which orders it "
            f"among them; these have none: {missing}")
    names = [name for name, _ in galaxies]
    items = [tuple(item) for item in depth_order]
    listed = [name for item in items for name in item]
    if unknown := [name for name in listed if name not in names]:
        raise ValueError(
            f"the depth order names unknown galaxies {unknown}; the "
            f"galaxies are {names}")
    if len(set(listed)) != len(listed):
        raise ValueError(f"the depth order repeats galaxies: {listed}")
    # The members, galaxies then lenses, by redshift: ('galaxy' or 'lens',
    # their index, their redshift)
    members = [('galaxy', i, z) for i, (_, z) in enumerate(galaxies)] \
        + [('lens', i, z) for i, (_, z) in enumerate(lenses)]
    members.sort(key=lambda member: member[2])
    runs = [[members[0]]]
    for previous, member in zip(members, members[1:]):
        if velocity_separation(previous[2], member[2]) \
                <= PLANE_LINKING_VELOCITY:
            runs[-1].append(member)
        else:
            runs.append([member])
    planes = tuple(_plane(run) for run in runs)
    _check_groups(items, galaxies, planes)
    _check_order(items, names, planes)
    return planes


def _plane(run: list[tuple[str, int, float]]) -> Plane:
    """
    Return the plane of members linked into one (see plane_layout): at
    the mean redshift of its lenses, or of its galaxies without lenses.
    """
    galaxies = tuple(sorted(i for kind, i, _ in run if kind == 'galaxy'))
    lenses = tuple(sorted(i for kind, i, _ in run if kind == 'lens'))
    redshifts = [z for kind, _, z in run if kind == 'lens'] \
        or [z for _, _, z in run]
    return Plane(sum(redshifts) / len(redshifts), galaxies, lenses)


def _check_groups(
        items: list[tuple[str, ...]],
        galaxies: Sequence[tuple[str | None, float | None]],
        planes: tuple[Plane, ...]
) -> None:
    """
    Raise ValueError if a group of the depth order (galaxies that pass
    through each other, at one distance) has galaxies of several planes:
    their redshifts put them apart (or one is mistyped).
    """
    names = [name for name, _ in galaxies]
    plane_of = {g: p for p, plane in enumerate(planes) for g in plane.galaxies}
    for item in items:
        indices = [names.index(name) for name in item]
        if len({plane_of[g] for g in indices}) > 1:
            redshifts = {names[g]: galaxies[g][1] for g in indices}
            raise ValueError(
                f"the galaxies of a group of the depth order pass through "
                f"each other, at one distance, but their redshifts "
                f"({redshifts}) put them in different planes, more than "
                f"{PLANE_LINKING_VELOCITY:g} km/s apart: give them "
                f"redshifts within {PLANE_LINKING_VELOCITY:g} km/s, or "
                f"split the group")


def _check_order(
        items: list[tuple[str, ...]],
        names: list[str | None],
        planes: tuple[Plane, ...]
) -> None:
    """
    Raise ValueError if the depth order puts a galaxy in front of one of
    a nearer plane: it orders the galaxies of a plane, and the redshifts
    the planes.
    """
    plane_of = {g: p for p, plane in enumerate(planes) for g in plane.galaxies}
    # (the galaxy of the farthest plane so far)
    farthest = None
    for item in items:
        p = plane_of[names.index(item[0])]
        if farthest is not None and p < farthest[1]:
            raise ValueError(
                f"the depth order puts galaxy {farthest[0]!r} in front of "
                f"{item[0]!r}, which its redshift puts in front of it: "
                f"the depth order orders the galaxies of one plane, and "
                f"the redshifts the planes")
        if farthest is None or p > farthest[1]:
            farthest = (item[0], p)
