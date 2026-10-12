"""
What an observation sees: the galaxies, the components of each whose
light it sees, and its line of sight (Scene); and their evaluation on the
grid of the observation (ScenePlan).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.model import Band, BandEmission, DustOpacity, Emission, Galaxy
from gbkfit.model.components._component_set import (
    IMAGE_SPECTRAL_AXIS, ComponentSetPlan)
from gbkfit.model.components._transfer import TransferPlan
from gbkfit.params import ParamValues
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from .line_of_sight import LineOfSight


__all__ = [
    'Scene',
    'ScenePlan',
    'make_scene'
]


class Scene:
    """
    What an observation sees: galaxies, the indices of the components of
    each whose light it sees (its emitters; the opacity of all components
    applies), its emission lines (all if None) or its band (if known), and
    its line of sight. The parameters of each galaxy have the names of the
    mapping of the galaxy among those of the observation group, and its
    extra outputs the prefix of the galaxy.
    """

    def __init__(
            self,
            galaxies: Sequence[Galaxy],
            mappings: Sequence[dict[str, str]],
            prefixes: Sequence[str],
            emitters: Sequence[tuple[int, ...]],
            lines: tuple[str, ...] | None,
            band: Band | None,
            line_of_sight: LineOfSight
    ):
        self._galaxies = tuple(galaxies)
        self._mappings = tuple(mappings)
        self._prefixes = tuple(prefixes)
        self._emitters = tuple(emitters)
        self._lines = lines
        self._band = band
        self._line_of_sight = line_of_sight

    def radius(self) -> float:
        """
        Return the outer radius (arcsec) of the largest disk it evaluates:
        of the components whose light it sees, and those with dust; 0
        without disks.
        """
        radii = [
            cmp.density_disk().rnodes()[-1]
            for galaxy, emitters in zip(self._galaxies, self._emitters)
            for i, cmp in enumerate(galaxy.components())
            if (i in emitters or isinstance(cmp.opacity(), DustOpacity))
            and cmp.density_disk() is not None]
        return max(radii, default=0.0)

    def component_params(self, name: str) -> dict[str, str]:
        """
        Return the names of the parameters of the component of the given
        name ('galaxy.component' when there are several galaxies) among
        those of the observation group, by its own names. Raise
        ConfigError if no component has the name.
        """
        g, i = _named_components(self._galaxies, [name])[name]
        mapping = self._mappings[g]
        return {
            own: mapping[full]
            for own, full in self._galaxies[g].component_params(i).items()}

    def plan(
            self,
            driver: Driver,
            grid: gridutils.Grid,
            dtype: np.dtype
    ) -> 'ScenePlan':
        """
        Plan the evaluation of the scene on a driver and dtype, on the
        grid of an image (no spectral axis) or a spectral cube.
        """
        return ScenePlan(self, driver, grid, dtype)


class ScenePlan:
    """
    The evaluation of a scene on a grid: the plan of each galaxy, whose
    light adds up in the image or spectral cube.
    """

    def __init__(
            self,
            scene: Scene,
            driver: Driver,
            grid: gridutils.Grid,
            dtype: np.dtype
    ):
        image = grid.spectral_axis is None
        spatial = grid if image else grid.spatial()
        spectral = IMAGE_SPECTRAL_AXIS if image else grid.spectral()
        depth = scene._line_of_sight.depth_axis(spatial, scene.radius())
        self._key = 'image' if image else 'scube'
        self._scene = scene
        self._plans: tuple[ComponentSetPlan | TransferPlan, ...] = tuple(
            scene._line_of_sight.plan_galaxy(
                galaxy, driver, spatial, spectral, depth, dtype, emitters,
                scene._lines, scene._band)
            for galaxy, emitters in zip(scene._galaxies, scene._emitters))

    def evaluate(
            self,
            params: ParamValues,
            data: DeviceArray,
            out_extra: dict[str, Any] | None
    ) -> None:
        """
        Add the light of the galaxies to the data (the image or spectral
        cube), with the parameters of the observation group (params). The
        extra outputs of each galaxy have its prefix.
        """
        scene = self._scene
        for plan, mapping, prefix in zip(
                self._plans, scene._mappings, scene._prefixes):
            extra = {} if out_extra is not None else None
            plan.evaluate(
                {name: params[full] for name, full in mapping.items()},
                {self._key: data}, extra)
            if out_extra is not None:
                out_extra.update(
                    {f'{prefix}{key}': value for key, value in extra.items()})


def make_scene(
        galaxies: Sequence[Galaxy],
        mappings: Sequence[dict[str, str]],
        prefixes: Sequence[str],
        components: Sequence[str] | None,
        lines: tuple[str, ...] | None,
        band: Band | None,
        line_of_sight: LineOfSight,
        emission: type[Emission]
) -> Scene:
    """
    Return the scene of an observation of the given galaxies (see Scene),
    which sees the emission of the given type (see Observable.emission),
    in the given band if known (an image). The names of the components it
    sees (all those with that emission, in its band, if None) are
    qualified by the names of their galaxies ('galaxy.component') when it
    sees several.

    Raises
    ------
    ConfigError
        If a name is unknown or repeats, a component does not emit what it
        sees, it sees nothing, a line is not one of the components it
        sees, a component cannot be seen along its line of sight (see
        LineOfSight.check), or it sees several galaxies and one has
        opacity (their order along the line of sight is not supported
        yet).
    """
    kind = "an image, which sees the components that emit in a band" \
        if emission is BandEmission else \
        "a spectral cube, which sees the components that emit lines"
    emitters = _select_emitters(galaxies, components, emission, band, kind)
    if not any(emitters):
        raise ConfigError(f"the observation is {kind}; it sees none")
    _check_lines(galaxies, emitters, lines)
    absorbers = [
        [i for i, cmp in enumerate(galaxy.components())
         if isinstance(cmp.opacity(), DustOpacity)]
        for galaxy in galaxies]
    if len(galaxies) > 1 and any(
            cmp.opacity() is not None
            for galaxy in galaxies for cmp in galaxy.components()):
        raise ConfigError(
            "an observation of several galaxies with opacity is not "
            "supported yet (it needs their order along the line of sight)")
    for galaxy, galaxy_emitters, galaxy_absorbers in zip(
            galaxies, emitters, absorbers):
        for i in sorted(set(galaxy_emitters) | set(galaxy_absorbers)):
            component = galaxy.components()[i]
            try:
                line_of_sight.check(component)
            except ConfigError as e:
                raise ConfigError(
                    f"component {_label(galaxy, i)}: {e}") from e
    return Scene(
        galaxies, mappings, prefixes, [tuple(e) for e in emitters], lines,
        band, line_of_sight)


def _label(galaxy: Galaxy, index: int) -> str:
    """Return the label of a component in messages."""
    name = galaxy.components()[index].name()
    label = repr(name) if name is not None else str(index)
    return label if galaxy.name() is None \
        else f"{label} of galaxy {galaxy.name()!r}"


def _select_emitters(
        galaxies: Sequence[Galaxy],
        names: Sequence[str] | None,
        emission: type,
        band: Band | None,
        kind: str
) -> list[list[int]]:
    """
    Return the indices of the components of each galaxy whose light is
    seen: those of the given names (qualified when there are several
    galaxies), or all those whose emission is of the given type, in the
    given band (see Emission.emits_in), in their order. kind describes the
    observation in messages.
    """
    def seen(cmp):
        return isinstance(cmp.emission(), emission) \
            and cmp.emission().emits_in(band)
    if names is None:
        return [
            [i for i, cmp in enumerate(galaxy.components()) if seen(cmp)]
            for galaxy in galaxies]
    if not names:
        raise ConfigError("at least one component must be selected")
    if len(set(names)) != len(names):
        raise ConfigError(f"the selected components repeat names: {names}")
    known = _named_components(galaxies, names)
    selected = [[] for _ in galaxies]
    for name in names:
        g, i = known[name]
        if not isinstance(galaxies[g].components()[i].emission(), emission):
            raise ConfigError(
                f"the component {name!r} is selected, but the observation "
                f"is {kind}")
        if not seen(galaxies[g].components()[i]):
            raise ConfigError(
                f"the component {name!r} is selected, but it does not emit "
                f"in the band {band.name()!r} of the image")
        selected[g].append(i)
    return [sorted(indices) for indices in selected]


def _named_components(
        galaxies: Sequence[Galaxy], names: Sequence[str]
) -> dict[str, tuple[int, int]]:
    """
    Return the indices of the galaxy and of the component of each named
    component of the galaxies, by its name ('galaxy.component' when there
    are several galaxies). Raise ConfigError unless the given names are
    among them.
    """
    known = {}
    for g, galaxy in enumerate(galaxies):
        for i, cmp in enumerate(galaxy.components()):
            if cmp.name() is None:
                continue
            key = cmp.name() if len(galaxies) == 1 \
                else f'{galaxy.name()}.{cmp.name()}'
            known[key] = (g, i)
    if unknown := [name for name in names if name not in known]:
        raise ConfigError(
            f"there are no components named {unknown}; the named components "
            f"are {list(known)}")
    return known


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
