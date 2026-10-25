"""
What an observation sees: the galaxies it sees (SeenGalaxies), and the
scene of the components whose light it sees, checked along its line of
sight (make_scene; see gbkfit.model.Scene).
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

from gbkfit.model import (
    STARS, Band, BandEmission, DataUnit, DepthUnit, DustOpacity, Emission,
    Galaxy, LineEmission, LineOpacity, Scene, StarsEmission,
    named_components)
from gbkfit.utils import parseutils, unitutils
from gbkfit.utils.parseutils import ConfigError

if TYPE_CHECKING:
    from .line_of_sight import LineOfSight


__all__ = [
    'SeenGalaxies',
    'check_component_names',
    'make_scene'
]


class SeenGalaxies(NamedTuple):
    """
    The galaxies an observation sees (see ObservationGroup): the
    galaxies, the mapping of the parameters of each to those of the
    observation group, their labels (the id 'galaxy' of their extra
    outputs), their order along the line of sight, front to back (in
    units, see DepthUnit; None if not given), and the RA and Dec (degrees)
    of the origin of the frame of their parameters (that of the first
    observation on the sky; None without one).
    """
    galaxies: tuple[Galaxy, ...]
    mappings: tuple[dict[str, str], ...]
    labels: tuple[str, ...]
    order: tuple[DepthUnit, ...] | None
    sky_origin: tuple[float, float] | None


def make_scene(
        seen: SeenGalaxies,
        components: Sequence[str] | None,
        lines: tuple[str, ...] | None,
        band: Band | None,
        line_of_sight: 'LineOfSight',
        emission: type[Emission],
        data_unit: DataUnit | None = None
) -> Scene:
    """
    Return the scene of an observation of the galaxies it sees (see
    Scene), along its line of sight, which sees the emission of the given
    type (see Observable.emission), in the given band if known (an image),
    and of the unit of its data, if any (see Observable.data_unit). The
    names of the components it sees (all those with that emission, in its
    band, if None) are qualified by the names of their galaxies
    ('galaxy.component') when it sees several (see named_components). A
    spectral cube sees the light of stars (see StarsEmission) too when its
    lines name it (STARS).

    Raises
    ------
    ConfigError
        If a name is unknown or repeats, a component does not emit what it
        sees, it sees nothing, a line is not one of the components it
        sees, a component cannot be seen along its line of sight (see
        LineOfSight.check), it sees several galaxies with opacity that its
        line of sight cannot order, or not in that order (see
        LineOfSight.check_order), or without data of light, it sees
        components of different units (its model is then in the unit of
        its components).
    """
    galaxies, order = seen.galaxies, seen.order
    kind = "an image, which sees the components that emit in a band" \
        if emission is BandEmission else \
        "a spectral cube, which sees the components that emit lines (and " \
        f"stars, when its lines name {STARS!r})"
    # (the types of emission it sees: a cube whose lines name the stars
    # sees them, and only them if they are all it names)
    types = (emission,)
    if emission is LineEmission and lines is not None and STARS in lines:
        types = (StarsEmission,) if set(lines) == {STARS} \
            else (LineEmission, StarsEmission)
    emitters = _select_emitters(galaxies, components, types, band, kind)
    if not any(emitters):
        raise ConfigError(f"the observation is {kind}; it sees none")
    _check_lines(galaxies, emitters, lines)
    if data_unit is None:
        _check_one_unit(galaxies, emitters)
    absorbers = [
        [i for i, cmp in enumerate(galaxy.components())
         if isinstance(cmp.opacity(), DustOpacity)]
        for galaxy in galaxies]
    if len(galaxies) > 1 and any(
            _has_opacity(galaxy, galaxy_emitters)
            for galaxy, galaxy_emitters in zip(galaxies, emitters)):
        line_of_sight.check_order(galaxies, order)
    for galaxy, galaxy_emitters, galaxy_absorbers in zip(
            galaxies, emitters, absorbers):
        for i in sorted(set(galaxy_emitters) | set(galaxy_absorbers)):
            component = galaxy.components()[i]
            try:
                line_of_sight.check(component, emission)
            except ConfigError as e:
                raise ConfigError(
                    f"component {_label(galaxy, i)}: {e}") from e
    return Scene(
        galaxies, seen.mappings, seen.labels, [tuple(e) for e in emitters],
        lines, band, order, data_unit)


def _has_opacity(galaxy: Galaxy, emitters: Sequence[int]) -> bool:
    """
    Return whether a galaxy dims light in an observation that sees the
    given emitters of it: whether it has components with dust, or among
    those emitters, components with line opacity.
    """
    return any(
        isinstance(cmp.opacity(), DustOpacity)
        or (i in emitters and isinstance(cmp.opacity(), LineOpacity))
        for i, cmp in enumerate(galaxy.components()))


def _label(galaxy: Galaxy, index: int) -> str:
    """Return the label of a component in messages."""
    name = galaxy.components()[index].name()
    label = repr(name) if name is not None else str(index)
    return label if galaxy.name() is None \
        else f"{label} of galaxy {galaxy.name()!r}"


def _select_emitters(
        galaxies: Sequence[Galaxy],
        names: Sequence[str] | None,
        types: tuple[type[Emission], ...],
        band: Band | None,
        kind: str
) -> list[list[int]]:
    """
    Return the indices of the components of each galaxy whose light is
    seen: those of the given names (qualified when there are several
    galaxies), or all those whose emission is of one of the given types,
    in the given band (see Emission.emits_in), in their order. kind
    describes the observation in messages.
    """
    def seen(cmp):
        return isinstance(cmp.emission(), types) \
            and cmp.emission().emits_in(band)
    if names is None:
        return [
            [i for i, cmp in enumerate(galaxy.components()) if seen(cmp)]
            for galaxy in galaxies]
    if not names:
        raise ConfigError("at least one component must be selected")
    if len(set(names)) != len(names):
        raise ConfigError(f"the selected components repeat names: {names}")
    known = named_components(galaxies, names)
    selected = [[] for _ in galaxies]
    for name in names:
        g, i = known[name]
        if not isinstance(galaxies[g].components()[i].emission(), types):
            raise ConfigError(
                f"the component {name!r} is selected, but the observation "
                f"is {kind}")
        if not seen(galaxies[g].components()[i]):
            raise ConfigError(
                f"the component {name!r} is selected, but it does not emit "
                f"in the band {band.name()!r} of the image")
        selected[g].append(i)
    return [sorted(indices) for indices in selected]


def _check_lines(
        galaxies: Sequence[Galaxy],
        emitters: Sequence[Sequence[int]],
        names: tuple[str, ...] | None
) -> None:
    """
    Raise ConfigError unless each of the names of lines (None for all) is
    a line of one of the components seen.
    """
    if names is None:
        return
    known = set()
    for galaxy, indices in zip(galaxies, emitters):
        for i in indices:
            known |= set(galaxy.components()[i].line_names())
    if unknown := [name for name in names if name not in known]:
        raise ConfigError(
            f"no component has the lines {unknown}; the lines of the "
            f"components are {sorted(known)}")


def _check_one_unit(
        galaxies: Sequence[Galaxy], emitters: Sequence[Sequence[int]]
) -> None:
    """
    Raise ConfigError if the emitters of the galaxies (by index) have
    light of different units: without data of light (no data, or e.g.
    only velocity moments), the model of an observation is in the unit
    of the components it sees (and of none, if they have none).
    """
    units = {
        unitutils.format_unit(unit): _label(galaxy, i)
        for galaxy, indices in zip(galaxies, emitters) for i in indices
        if (unit := galaxy.components()[i].emission().unit()) is not None}
    if len(units) > 1:
        raise ConfigError(
            f"without data of light (no data, or only velocity moments), "
            f"the model of an observation is in the unit of the components "
            f"it sees, which have several: "
            f"{', '.join(f'{u!r} ({c})' for u, c in units.items())}; give "
            f"the observation data of light in a unit, or see these "
            f"components in separate observations")


def check_component_names(names: Sequence[str]) -> None:
    """
    Raise ConfigError unless each name is the name of a component, or that
    of a galaxy and of one of its components ('galaxy.component').
    """
    for name in names:
        parts = name.split('.') if isinstance(name, str) else [name]
        if len(parts) > 2:
            raise ConfigError(
                f"a component is named 'component' or 'galaxy.component'; "
                f"{name!r} is not")
        for part in parts:
            parseutils.check_name(part)
