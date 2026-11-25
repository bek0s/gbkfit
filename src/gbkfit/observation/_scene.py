"""
The scenes of an observation on the sky (see ObservationSky): the
components whose light it sees, of the galaxies it sees (see
SeenGalaxies), drawn in a scene for each of their planes (make_scenes;
see gbkfit.model.Scene), which decide how each is evaluated along the
line of sight.
"""

from collections.abc import Sequence
from typing import NamedTuple

import gbkfit.units
from gbkfit.model import (
    Component, DataUnit, DepthUnit, Galaxy, Scene, SceneEvaluation, View,
    check_data_unit)
from .observation import SeenGalaxies, SeenPlane


__all__ = [
    'SeenScenes',
    'make_scenes'
]


class SeenScenes(NamedTuple):
    """
    What an observation on the sky sees of the galaxies it sees (see
    make_scenes): the galaxies, the planes it draws, near to far (see
    SeenPlane), the scene of each, and the indices of the components of
    each galaxy whose light it sees.
    """
    seen: SeenGalaxies
    planes: tuple[SeenPlane, ...]
    scenes: tuple[Scene, ...]
    emitters: tuple[tuple[int, ...], ...]

    def lensed(self) -> bool:
        """
        Check whether a lens bends the light of a plane it draws: it then
        draws several.
        """
        return any(plane.traced is not None for plane in self.planes)

    def scene(self) -> Scene:
        """
        Return the one scene of an observation whose light no lens bends
        (see lensed): that of its one plane.

        Raises
        ------
        ValueError
            If a lens bends the light of a plane.
        """
        if self.lensed():
            raise ValueError(
                "the observation sees planes behind lenses, each a scene of "
                "its own")
        return self.scenes[0]

    def view(self) -> View:
        """Return what it sees of the light of the components."""
        return self.scenes[0].view()

    def data_unit(self) -> DataUnit | None:
        """
        Return the unit of its data, which the light of the components
        takes (None without data).
        """
        return self.scenes[0].data_unit()

    def components(self) -> list[tuple[int, int, Component]]:
        """
        Return the components whose light it sees, each with the index of
        its galaxy (among those it sees) and its own index.
        """
        return [
            (g, i, self.seen.galaxies[g].components()[i])
            for g, emitters in enumerate(self.emitters) for i in emitters]

    def orbit_emitters(self) -> tuple[tuple[int, int], ...]:
        """
        Return the components whose light it sees and whose stars move on
        orbits whose weights are solved (see Scene.orbit_emitters), each
        as the index of its galaxy (among those it sees) and its own
        index.
        """
        return tuple(
            (plane.galaxies[g], i)
            for plane, scene in zip(self.planes, self.scenes)
            for g, i in scene.orbit_emitters())


def make_scenes(
        seen: SeenGalaxies,
        view: View,
        data_unit: DataUnit | None = None
) -> SeenScenes:
    """
    Return the scenes of an observation of the galaxies it sees (see
    SeenScenes): a scene of the galaxies of each of their planes (see
    Scene and SeenPlane), of what it sees (see View: its tracer, on a
    spectral axis or in an image; see ObservationSky.view), and of the
    unit of its data, if any (see ObservationSky.data_unit). It sees the
    light of the components of each galaxy that the group chose for it
    (see SeenGalaxies), or of all those whose light its tracer traces
    (see Tracer.sees). The opacity of every component of the galaxies
    applies.

    It draws the planes whose galaxies emit what it sees, and those in
    front of them with opacity, which dims their light: every plane but
    the farthest has light from behind it (see Scene.behind).

    Raises
    ------
    ValueError
        If a chosen component does not emit what it sees, a component of
        what it sees would be seen only in part (see Tracer.sees), it
        sees nothing, its tracer cannot see what it sees (see
        Tracer.check; e.g. a line that none of the components it sees
        has), it sees a continuum with a unit of a galaxy whose redshift
        is not known (see _check_redshifts), it sees none of the light of
        galaxies behind a lens (nor of any behind them) and no other plane
        behind a lens, it sees several
        galaxies with opacity in a plane (which radiative transfer marches
        one after the other, see Scene.evaluation) that the order of the
        galaxies along the line of sight does not all place (see
        ObservationGroup: depth_order), or without data of light, it sees
        components of different units (its model is then in the unit of
        its components), or with data without a unit, a component whose
        light has one (see check_data_unit).
    """
    galaxies = seen.galaxies
    tracer = view.tracer
    kind = f"of the tracer {tracer.type()!r}"
    emitters = _select_emitters(seen, view, kind)
    if not any(emitters):
        # (why, if its tracer can say it: e.g. a line no component has)
        tracer.check([
            component.light() for galaxy in galaxies
            for component in galaxy.components()
            if component.light() is not None])
        raise ValueError(
            f"the observation is {kind}; it sees none of the components")
    tracer.check([
        galaxy.components()[i].light()
        for galaxy, indices in zip(galaxies, emitters) for i in indices])
    _check_redshifts(galaxies, emitters, view)
    if data_unit is None:
        _check_one_unit(galaxies, emitters)
    # (here, before any plan: the copies of an observation with other
    # data, which can share its plan, are checked again, see
    # Observation.same_plan)
    for galaxy, galaxy_emitters in zip(galaxies, emitters):
        for i in galaxy_emitters:
            component = galaxy.components()[i]
            try:
                check_data_unit(component.light().unit(), data_unit)
            except ValueError as e:
                raise ValueError(
                    f"component {_label(galaxy, i)}: {e}") from e
    # The planes drawn, far to near, each with whether one is behind it
    drawn = []
    for plane in reversed(seen.planes):
        behind = bool(drawn)
        if any(emitters[g] for g in plane.galaxies) or behind and any(
                galaxies[g].absorbers() for g in plane.galaxies):
            drawn.append((plane, behind))
    drawn_planes = [plane for plane, _ in drawn]
    left = [
        galaxies[g].name() for plane in seen.planes
        if plane not in drawn_planes for g in plane.galaxies]
    # (the one scene of an image without planes behind lenses takes the
    # states of all the galaxies the observation sees)
    if left and all(plane.traced is None for plane in drawn_planes):
        raise ValueError(
            f"the observation sees none of the light of the galaxies {left} "
            f"behind a lens, nor any behind them: leave them out of what it "
            f"sees (the section seen of its setup)")
    scenes = []
    for plane, behind in reversed(drawn):
        plane_galaxies = [galaxies[g] for g in plane.galaxies]
        plane_emitters = [tuple(emitters[g]) for g in plane.galaxies]
        scene = Scene(
            plane_galaxies, [seen.mappings[g] for g in plane.galaxies],
            [seen.labels[g] for g in plane.galaxies], plane_emitters, view,
            plane.order, data_unit, behind,
            lensed=plane.traced is not None)
        if len(plane_galaxies) > 1 \
                and scene.evaluation() is SceneEvaluation.TRANSFER:
            _check_order(plane_galaxies, plane.order)
        scenes.append(scene)
    return SeenScenes(
        seen, tuple(reversed(drawn_planes)), tuple(scenes),
        tuple(tuple(e) for e in emitters))


def _check_order(
        galaxies: Sequence[Galaxy], order: Sequence[DepthUnit] | None
) -> None:
    """
    Raise ValueError unless the order of the galaxies of a plane along the
    line of sight (in units, see ObservationGroup: depth_order) places
    every one of them: radiative transfer marches galaxies with opacity
    one after the other, front to back.
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
        seen: SeenGalaxies, view: View, kind: str
) -> list[list[int]]:
    """
    Return the indices of the components of each galaxy seen whose light
    is seen: those that the group chose (see SeenGalaxies), or all those
    whose light the tracer of the view traces (see Tracer.sees), in their
    order. kind describes the observation in messages.
    """
    def sees(galaxy: Galaxy, i: int) -> bool:
        cmp = galaxy.components()[i]
        if cmp.light() is None:
            return False
        try:
            return view.sees(cmp.light(), cmp.moves())
        except ValueError as e:
            raise ValueError(
                f"the observation is {kind}; component "
                f"{_label(galaxy, i)}: {e}") from e
    selected = []
    for galaxy, chosen in zip(seen.galaxies, seen.components):
        if chosen is None:
            selected.append([
                i for i in range(len(galaxy.components()))
                if sees(galaxy, i)])
            continue
        for i in chosen:
            if not sees(galaxy, i):
                raise ValueError(
                    f"the setup chose component {_label(galaxy, i)} for "
                    f"the observation, which is {kind}: it does not see it")
        selected.append(list(chosen))
    return selected


def _check_redshifts(
        galaxies: Sequence[Galaxy],
        emitters: Sequence[Sequence[int]],
        view: View
) -> None:
    """
    Raise ValueError if the tracer of the view sees the continuum of a
    component dimmed by the redshift of its galaxy (see
    Tracer.dims_by_redshift; its light in a unit, as the galaxy emits it),
    and neither the galaxy gives its redshift nor the component has a
    systemic velocity: the redshift is then not known (see
    _units.BandFactors).
    """
    if not view.tracer.dims_by_redshift():
        return
    for galaxy, indices in zip(galaxies, emitters):
        if galaxy.redshift_given():
            continue
        for i in indices:
            component = galaxy.components()[i]
            if component.light().unit() is not None \
                    and component.frame_params().vsys is None:
                raise ValueError(
                    f"the light of component {_label(galaxy, i)} is in a "
                    f"unit, as the galaxy emits it, which the observation "
                    f"sees dimmed by its redshift: give the galaxy its "
                    f"redshift (the option redshift; 0 for none)")


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
        if (unit := galaxy.components()[i].light().unit()) is not None}
    if len(units) > 1:
        raise ValueError(
            f"without data of light (no data, or only velocity moments), "
            f"the model of an observation is in the unit of the components "
            f"it sees, which have several: "
            f"{', '.join(f'{u!r} ({c})' for u, c in units.items())}; give "
            f"the observation data of light in a unit, or see these "
            f"components in separate observations")
