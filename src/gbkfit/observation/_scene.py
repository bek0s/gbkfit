"""
The scene of an observation on the sky (see ObservationSky): the
components whose light it sees, of the galaxies it sees (see
SeenGalaxies), which decide how it is evaluated along the line of sight
(make_scene; see gbkfit.model.Scene).
"""

from collections.abc import Sequence

import gbkfit.units
from gbkfit.config import check_name
from gbkfit.model import (
    Component, DataUnit, DepthUnit, Galaxy, Scene, SceneEvaluation, View,
    check_data_unit, named_components)
from .observation import SeenGalaxies


__all__ = [
    'check_component_names',
    'make_scene'
]


def make_scene(
        seen: SeenGalaxies,
        components: Sequence[str] | None,
        view: View,
        data_unit: DataUnit | None = None
) -> Scene:
    """
    Return the scene of an observation of the galaxies it sees (see
    Scene), of what it sees (see View: its band, or its spectral axis with
    its lines and continuum; see ObservationSky.view), and of the unit of
    its data, if any (see ObservationSky.data_unit). The
    names of the components it sees (all those whose emission the view
    sees, if None; see Emission.seen_in) are qualified by the names of
    their galaxies ('galaxy.component') when it sees several (see
    named_components). A spectral view of the continuum sees the
    velocities of the light in bands of the components that move (e.g.
    the kinematics of stars).

    Raises
    ------
    ValueError
        If a name is unknown or repeats, a component does not emit what it
        sees, it sees nothing, a line is not one of the components it
        sees, it sees several galaxies with opacity
        (which radiative transfer marches one after the other, see
        Scene.evaluation) that the order of the galaxies along the line of
        sight does not all place (see ObservationGroup: depth_order), or
        without data of light, it sees components of different units (its
        model is then in the unit of its components), or with data
        without a unit, a component whose light has one (see
        check_data_unit).
    """
    galaxies, order = seen.galaxies, seen.order
    kind = "an image, which sees the components that emit in a band" \
        if not view.spectral else \
        "a spectral cube, which sees the components that emit lines" + (
            " and the continuum of those that emit in bands and move"
            if view.continuum else "")
    emitters = _select_emitters(galaxies, components, view, kind)
    if not any(emitters):
        raise ValueError(f"the observation is {kind}; it sees none")
    _check_lines(galaxies, emitters, view.lines)
    if data_unit is None:
        _check_one_unit(galaxies, emitters)
    # (here, before any plan: the copies of an observation with other
    # data, which can share its plan, are checked again, see
    # Observation.same_plan)
    for galaxy, galaxy_emitters in zip(galaxies, emitters):
        for i in galaxy_emitters:
            component = galaxy.components()[i]
            try:
                check_data_unit(component.emission().unit(), data_unit)
            except ValueError as e:
                raise ValueError(
                    f"component {_label(galaxy, i)}: {e}") from e
    scene = Scene(
        galaxies, seen.mappings, seen.labels, [tuple(e) for e in emitters],
        view, order, data_unit)
    if len(galaxies) > 1 \
            and scene.evaluation() is SceneEvaluation.TRANSFER:
        _check_order(galaxies, order)
    return scene


def _check_order(
        galaxies: Sequence[Galaxy], order: Sequence[DepthUnit] | None
) -> None:
    """
    Raise ValueError unless the order of the galaxies along the line of
    sight (in units, see ObservationGroup: depth_order) places every one
    of them: radiative transfer marches galaxies with opacity one after
    the other, front to back.
    """
    ordered = {g for unit in order or () for g in unit.galaxies}
    if missing := [
            galaxy.name() for g, galaxy in enumerate(galaxies)
            if g not in ordered]:
        raise ValueError(
            f"an observation of several galaxies with opacity marches them "
            f"by radiative transfer, one after the other, front to back: "
            f"the depth_order of the observation group must place the "
            f"galaxies {missing}")


def _label(galaxy: Galaxy, index: int) -> str:
    """Return the label of a component in messages."""
    name = galaxy.components()[index].name()
    label = repr(name) if name is not None else str(index)
    return label if galaxy.name() is None \
        else f"{label} of galaxy {galaxy.name()!r}"


def _select_emitters(
        galaxies: Sequence[Galaxy],
        names: Sequence[str] | None,
        view: View,
        kind: str
) -> list[list[int]]:
    """
    Return the indices of the components of each galaxy whose light is
    seen: those of the given names (qualified when there are several
    galaxies), or all those whose emission the view sees (see
    Emission.seen_in), in their order. kind describes the observation in
    messages.
    """
    def seen(cmp: Component) -> bool:
        return cmp.emission() is not None \
            and cmp.emission().seen_in(view, cmp.moves())
    if names is None:
        return [
            [i for i, cmp in enumerate(galaxy.components()) if seen(cmp)]
            for galaxy in galaxies]
    if not names:
        raise ValueError("at least one component must be selected")
    if len(set(names)) != len(names):
        raise ValueError(f"the selected components repeat names: {names}")
    known = named_components(galaxies, names)
    selected = [[] for _ in galaxies]
    for name in names:
        g, i = known[name]
        if not seen(galaxies[g].components()[i]):
            raise ValueError(
                f"the component {name!r} is selected, but the observation "
                f"is {kind}: it does not see it")
        selected[g].append(i)
    return [sorted(indices) for indices in selected]


def _check_lines(
        galaxies: Sequence[Galaxy],
        emitters: Sequence[Sequence[int]],
        names: tuple[str, ...] | None
) -> None:
    """
    Raise ValueError unless each of the names of lines (None for all) is
    a line of one of the components seen.
    """
    if names is None:
        return
    known = set()
    for galaxy, indices in zip(galaxies, emitters):
        for i in indices:
            known |= set(galaxy.components()[i].emission().line_names())
    if unknown := [name for name in names if name not in known]:
        raise ValueError(
            f"no component has the lines {unknown}; the lines of the "
            f"components are {sorted(known)}")


def _check_one_unit(
        galaxies: Sequence[Galaxy], emitters: Sequence[Sequence[int]]
) -> None:
    """
    Raise ValueError if the emitters of the galaxies (by index) have
    light of different units: without data of light (no data, or e.g.
    only velocity moments), the model of an observation is in the unit
    of the components it sees (and of none, if they have none).
    """
    units = {
        gbkfit.units.format_unit(unit): _label(galaxy, i)
        for galaxy, indices in zip(galaxies, emitters) for i in indices
        if (unit := galaxy.components()[i].emission().unit()) is not None}
    if len(units) > 1:
        raise ValueError(
            f"without data of light (no data, or only velocity moments), "
            f"the model of an observation is in the unit of the components "
            f"it sees, which have several: "
            f"{', '.join(f'{u!r} ({c})' for u, c in units.items())}; give "
            f"the observation data of light in a unit, or see these "
            f"components in separate observations")


def check_component_names(names: Sequence[str]) -> None:
    """
    Raise ValueError unless each name is the name of a component, or that
    of a galaxy and of one of its components ('galaxy.component').
    """
    for name in names:
        parts = name.split('.') if isinstance(name, str) else [name]
        if len(parts) > 2:
            raise ValueError(
                f"a component is named 'component' or 'galaxy.component'; "
                f"{name!r} is not")
        for part in parts:
            check_name(part)
