"""
What an observation sees: the galaxies, the components of each whose
light it sees, and its line of sight (Scene); and their evaluation on the
grid of the observation (ScenePlan).
"""

import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.model import (
    Band, BandEmission, DustOpacity, Emission, Galaxy, LineOpacity)
from gbkfit.model.components._component_plans import opacity_cubes
from gbkfit.model.components._component_set import (
    IMAGE_SPECTRAL_AXIS, ComponentSetPlan)
from gbkfit.model.components._transfer import (
    LineResolution, TransferPlan, warn_resolution)
from gbkfit.params import ParamValues
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError

if TYPE_CHECKING:
    from .line_of_sight import LineOfSight


__all__ = [
    'AdditiveScenePlan',
    'Scene',
    'ScenePlan',
    'TransferScenePlan',
    'make_scene'
]


class Scene:
    """
    What an observation sees: galaxies, the indices of the components of
    each whose light it sees (its emitters; the opacity of all components
    applies), its emission lines (all if None) or its band (if known), its
    line of sight, and the order of the galaxies along it, front to back
    (their indices; those it leaves out after them, in their order). The
    parameters of each galaxy have the names of the mapping of the galaxy
    among those of the observation group, and its extra outputs the prefix
    of the galaxy.
    """

    def __init__(
            self,
            galaxies: Sequence[Galaxy],
            mappings: Sequence[dict[str, str]],
            prefixes: Sequence[str],
            emitters: Sequence[tuple[int, ...]],
            lines: tuple[str, ...] | None,
            band: Band | None,
            line_of_sight: 'LineOfSight',
            order: Sequence[int] | None = None
    ):
        self._galaxies = tuple(galaxies)
        self._mappings = tuple(mappings)
        self._prefixes = tuple(prefixes)
        self._emitters = tuple(emitters)
        self._lines = lines
        self._band = band
        self._line_of_sight = line_of_sight
        order = tuple(order or ())
        self._order = order + tuple(
            g for g in range(len(self._galaxies)) if g not in order)

    def bounding_radius(self) -> float:
        """
        Return the bounding radius (arcsec) of the largest disk it
        evaluates (see _bounding_radius); 0 without disks.
        """
        return max(
            _bounding_radius(galaxy, emitters)
            for galaxy, emitters in zip(self._galaxies, self._emitters))

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
            dtype: np.dtype,
            smoothing: float | None
    ) -> 'ScenePlan':
        """
        Plan the evaluation of the scene on a driver and dtype, on the
        grid of an image (no spectral axis) or a spectral cube, along its
        line of sight (see LineOfSight.plan_scene), for an instrument that
        smooths the spectra of the model by the given dispersion (km/s;
        None for an image), which the check of the resolution of the light
        takes (see LineResolution).
        """
        return self._line_of_sight.plan_scene(
            self, driver, grid, dtype, smoothing)


class ScenePlan(abc.ABC):
    """
    The evaluation of a scene on a grid (see LineOfSight.plan_scene): the
    plan of each galaxy, whose light adds up in the image or spectral
    cube.
    """

    def __init__(self, scene: Scene, grid: gridutils.Grid):
        image = grid.spectral_axis is None
        self._scene = scene
        self._key = 'image' if image else 'scube'
        self._spatial = grid if image else grid.spatial()
        self._spectral = IMAGE_SPECTRAL_AXIS if image else grid.spectral()

    def evaluate(
            self,
            params: ParamValues,
            gravity: Sequence[dict[int, ParamValues]],
            data: DeviceArray,
            out_extra: dict[str, Any] | None
    ) -> None:
        """
        Add the light of the galaxies to the data (the image or spectral
        cube), with the parameters of the observation group (params) and
        what the gravity of each galaxy gives its components (gravity, see
        GravityPlan.evaluate). The extra outputs of each galaxy have its
        prefix.
        """
        scene = self._scene
        values = [
            {name: params[full] for name, full in mapping.items()}
            for mapping in scene._mappings]
        extras = [{} for _ in values] if out_extra is not None else None
        self._evaluate(values, gravity, {self._key: data}, extras)
        if out_extra is not None:
            for prefix, extra in zip(scene._prefixes, extras):
                out_extra.update(
                    {f'{prefix}{key}': value for key, value in extra.items()})

    @abc.abstractmethod
    def _evaluate(
            self,
            params: list[ParamValues],
            gravity: Sequence[dict[int, ParamValues]],
            outputs: dict[str, DeviceArray],
            extras: list[dict[str, Any]] | None
    ) -> None:
        """
        Add the light of the galaxies to the output array (see evaluate),
        with the parameters of each galaxy (by its own names) and what its
        gravity gives its components; with extras, the extra outputs of
        each galaxy go to its item.
        """
        pass


class AdditiveScenePlan(ScenePlan):
    """
    The evaluation of a scene whose galaxies add their light, each dimmed
    by its own opacity only (see Flat and Grid): on the depth axis of its
    line of sight, if it has one, for the largest disk of the scene. With
    extra outputs, it logs a warning if the depth axis resolves the
    velocity of the light too coarsely (see warn_resolution).
    """

    def __init__(
            self,
            scene: Scene,
            driver: Driver,
            grid: gridutils.Grid,
            dtype: np.dtype,
            smoothing: float | None
    ):
        super().__init__(scene, grid)
        depth = scene._line_of_sight.depth_axis(
            self._spatial, scene.bounding_radius())
        self._plans: tuple[ComponentSetPlan, ...] = tuple(
            galaxy.plan(
                driver, self._spatial, self._spectral, depth, dtype, emitters,
                opacity_cubes(
                    [galaxy], [emitters], scene._lines, scene._band,
                    dtype)[0],
                scene._lines, scene._band, smoothing)
            for galaxy, emitters in zip(scene._galaxies, scene._emitters))
        self._depth = depth

    def _evaluate(
            self,
            params: list[ParamValues],
            gravity: Sequence[dict[int, ParamValues]],
            outputs: dict[str, DeviceArray],
            extras: list[dict[str, Any]] | None
    ) -> None:
        resolutions = [
            plan.evaluate(
                params[g], gravity[g], outputs,
                extras[g] if extras is not None else None)
            for g, plan in enumerate(self._plans)]
        if extras is not None and self._depth is not None:
            warn_resolution(
                LineResolution(*np.max(resolutions, axis=0)),
                self._spectral.coords.step[0], 1, self._depth.step)


class TransferScenePlan(ScenePlan):
    """
    The evaluation of a scene by radiative transfer along each line of
    sight (see Transfer): each galaxy on a depth axis of its own (for its
    largest disk), marched one after the other, front to back, in the
    order of the scene, from the opacity of the galaxies in front of it
    (see TransferPlan). One set of cubes of opacity holds the dust of all
    of them (see opacity_cubes). (The light of galaxies without opacity
    just adds up, in any order.)

    With extra outputs, it logs a warning if the march resolves line
    opacity, or the depth axis the velocity of the light, too coarsely
    (see warn_resolution).
    """

    def __init__(
            self,
            scene: Scene,
            driver: Driver,
            grid: gridutils.Grid,
            dtype: np.dtype,
            slab: int | None,
            subchannels: int,
            smoothing: float | None
    ):
        super().__init__(scene, grid)
        spatial = self._spatial
        cubes = opacity_cubes(
            scene._galaxies, scene._emitters, scene._lines, scene._band,
            dtype)
        axes = [
            scene._line_of_sight.depth_axis(
                spatial, _bounding_radius(*galaxy))
            for galaxy in zip(scene._galaxies, scene._emitters)]
        self._plans: tuple[TransferPlan, ...] = tuple(
            galaxy.plan_transfer(
                driver, spatial, self._spectral, axis, dtype, emitters,
                galaxy_cubes, scene._lines, scene._band, slab, subchannels,
                smoothing)
            for galaxy, emitters, axis, galaxy_cubes in zip(
                scene._galaxies, scene._emitters, axes, cubes))
        self._subchannels = subchannels
        # (that of every galaxy)
        self._depth_step = axes[0].step
        # The state of the march from one galaxy to the next: the optical
        # depth of the dust in front, in each cube of opacity, and with
        # line opacity, that of the lines at each sub-channel
        self._driver = driver
        shape = (spatial.size[1], spatial.size[0])
        self._opacity_front = None
        if cubes[0].ncubes:
            self._opacity_front = driver.mem_alloc_d(
                (cubes[0].ncubes,) + shape, dtype)
        self._tau = None
        if any(plan.has_line_opacity() for plan in self._plans):
            self._tau = driver.mem_alloc_d(
                (self._spectral.size[0] * subchannels,) + shape, dtype)

    def _evaluate(
            self,
            params: list[ParamValues],
            gravity: Sequence[dict[int, ParamValues]],
            outputs: dict[str, DeviceArray],
            extras: list[dict[str, Any]] | None
    ) -> None:
        for state in (self._opacity_front, self._tau):
            if state is not None:
                self._driver.mem_fill(state, 0)
        # Each galaxy marches from the dust and the line opacity in front
        # of it, if any; with extras, its march tells how finely it
        # resolves its lines, and the worst gets a warning
        dust = line_opacity = False
        resolutions = []
        for g in self._scene._order:
            plan = self._plans[g]
            dust = dust or plan.has_dust()
            line_opacity = line_opacity or plan.has_line_opacity()
            resolution = plan.evaluate(
                params[g], gravity[g], outputs,
                extras[g] if extras is not None else None,
                self._opacity_front if dust else None,
                self._tau if line_opacity else None)
            if resolution is not None:
                resolutions.append(resolution)
        if resolutions:
            warn_resolution(
                LineResolution(*np.max(resolutions, axis=0)),
                self._spectral.coords.step[0], self._subchannels,
                self._depth_step)


def _bounding_radius(galaxy: Galaxy, emitters: Sequence[int]) -> float:
    """
    Return the radius (arcsec) of the sphere around the centre of a galaxy
    that holds all the density of the disks that an observation evaluates
    (of the given emitters, and the components with dust), at any
    inclination (see Disk.bounding_radius); 0 without disks.
    """
    radii = [
        cmp.density_disk().bounding_radius()
        for i, cmp in enumerate(galaxy.components())
        if (i in emitters or isinstance(cmp.opacity(), DustOpacity))
        and cmp.density_disk() is not None]
    return max(radii, default=0.0)


def make_scene(
        galaxies: Sequence[Galaxy],
        mappings: Sequence[dict[str, str]],
        prefixes: Sequence[str],
        components: Sequence[str] | None,
        lines: tuple[str, ...] | None,
        band: Band | None,
        line_of_sight: 'LineOfSight',
        emission: type[Emission],
        order: Sequence[int] | None = None
) -> Scene:
    """
    Return the scene of an observation of the given galaxies (see Scene),
    which sees the emission of the given type (see Observable.emission),
    in the given band if known (an image), and of their order along the
    line of sight, front to back (their indices, if given; see
    ObservationGroup: depth_order). The names of the components it
    sees (all those with that emission, in its band, if None) are
    qualified by the names of their galaxies ('galaxy.component') when it
    sees several.

    Raises
    ------
    ConfigError
        If a name is unknown or repeats, a component does not emit what it
        sees, it sees nothing, a line is not one of the components it
        sees, a component cannot be seen along its line of sight (see
        LineOfSight.check), or it sees several galaxies with opacity that
        its line of sight cannot order, or not in that order (see
        LineOfSight.check_order).
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
            _has_opacity(galaxy, galaxy_emitters)
            for galaxy, galaxy_emitters in zip(galaxies, emitters)):
        line_of_sight.check_order(galaxies, order)
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
        band, line_of_sight, order)


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
