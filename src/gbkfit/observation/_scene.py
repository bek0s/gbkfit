"""
What an observation sees: the galaxies, the components of each whose
light it sees, and its line of sight (Scene); and their evaluation on the
grid of the observation (ScenePlan).
"""

import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.model import (
    Band, BandEmission, DataUnit, DustOpacity, Emission, Extras, Galaxy,
    GalaxyState, LineOpacity)
from gbkfit.model.components._component_plans import opacity_cubes
from gbkfit.model.components._component_set import (
    IMAGE_SPECTRAL_AXIS, ComponentSetPlan)
from gbkfit.model.components.traits import RPTraitImage
from gbkfit.model.components._transfer import (
    LineResolution, TransferPlan, warn_resolution)
from gbkfit.params import ParamValues
from gbkfit.utils import gridutils, parseutils, unitutils
from gbkfit.utils.parseutils import ConfigError

if TYPE_CHECKING:
    from .line_of_sight import LineOfSight


__all__ = [
    'AdditiveScenePlan',
    'DepthUnit',
    'Scene',
    'ScenePlan',
    'SeenGalaxies',
    'TransferScenePlan',
    'make_scene'
]


class DepthUnit(NamedTuple):
    """
    Galaxies of a scene that radiative transfer marches together (see
    TransferScenePlan): one galaxy, or a group of galaxies that can pass
    through each other (see ObservationGroup: depth_order), by index, with
    the name of the group, and the parameter of the position of each along
    the line of sight, from the first of its group (its separation; None
    for the first, at 0).
    """
    galaxies: tuple[int, ...]
    name: str | None = None
    separations: tuple[str | None, ...] = (None,)


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


class Scene:
    """
    What an observation sees: galaxies, the indices of the components of
    each whose light it sees (its emitters; the opacity of all components
    applies), its emission lines (all if None) or its band (if known), its
    line of sight, the order of the galaxies along it, front to back (in
    units: galaxies or groups of them; those it leaves out after them, in
    their order), and the unit of its data, which the light of the
    emitters takes (see Component.plan; None without data). The
    parameters of each galaxy have the names of the mapping of the galaxy
    among those of the observation group, and its extra outputs its label
    as the id 'galaxy' (those of a group, its name as the id 'group').
    """

    def __init__(
            self,
            galaxies: Sequence[Galaxy],
            mappings: Sequence[dict[str, str]],
            labels: Sequence[str],
            emitters: Sequence[tuple[int, ...]],
            lines: tuple[str, ...] | None,
            band: Band | None,
            line_of_sight: 'LineOfSight',
            order: Sequence[DepthUnit] | None = None,
            data_unit: DataUnit | None = None
    ):
        self._galaxies = tuple(galaxies)
        self._mappings = tuple(mappings)
        self._labels = tuple(labels)
        self._emitters = tuple(emitters)
        self._lines = lines
        self._band = band
        self._line_of_sight = line_of_sight
        self._data_unit = data_unit
        order = tuple(order or ())
        ordered = {g for unit in order for g in unit.galaxies}
        self._order = order + tuple(
            DepthUnit((g,)) for g in range(len(self._galaxies))
            if g not in ordered)

    def bounding_radius(self) -> float:
        """
        Return the bounding radius (arcsec) of the largest component it
        evaluates (see _bounding_radius); 0 without extent.
        """
        return max(
            _bounding_radius(galaxy, emitters)
            for galaxy, emitters in zip(self._galaxies, self._emitters))

    def image_traits(self) -> tuple[RPTraitImage, ...]:
        """
        Return the image traits of the disks whose light it sees (see
        traits.RPTraitImage).
        """
        return tuple(
            trait
            for galaxy, emitters in zip(self._galaxies, self._emitters)
            for i in emitters
            if (disk := galaxy.components()[i].density_disk()) is not None
            for trait in disk.traits('rpt')
            if isinstance(trait, RPTraitImage))

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
            galaxies: Sequence[GalaxyState],
            data: DeviceArray,
            out_extra: Extras | None
    ) -> None:
        """
        Add the light of the galaxies to the data (the image or spectral
        cube), with the parameters of the observation group (params) and
        what each galaxy gives its components in the evaluation (see
        GalaxyState). The extra outputs of each galaxy have its label as
        the id 'galaxy', and those of a group of galaxies its name as the
        id 'group'.
        """
        scene = self._scene
        values = [
            {name: params[full] for name, full in mapping.items()}
            for mapping in scene._mappings]
        extras = None if out_extra is None else [
            out_extra.scoped(galaxy=label) for label in scene._labels]
        self._evaluate(
            params, values, [galaxy.given for galaxy in galaxies],
            {self._key: data},
            extras, out_extra)

    @abc.abstractmethod
    def _evaluate(
            self,
            group_params: ParamValues,
            params: list[ParamValues],
            given: Sequence[dict[int, ParamValues]],
            outputs: dict[str, DeviceArray],
            extras: list[Extras] | None,
            group_extras: Extras | None
    ) -> None:
        """
        Add the light of the galaxies to the output array (see evaluate),
        with the parameters of the observation group (group_params: e.g.
        the separations of the galaxies of a group, see DepthUnit), those
        of each galaxy (by its own names) and what it gives its components
        (given); with extras, the extra outputs of each galaxy go to
        its item, and those of each group of galaxies to group_extras,
        with its name as the id 'group'.
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
                scene._lines, scene._band, scene._data_unit, smoothing)
            for galaxy, emitters in zip(scene._galaxies, scene._emitters))
        self._depth = depth

    def _evaluate(
            self,
            group_params: ParamValues,
            params: list[ParamValues],
            given: Sequence[dict[int, ParamValues]],
            outputs: dict[str, DeviceArray],
            extras: list[Extras] | None,
            group_extras: Extras | None
    ) -> None:
        resolutions = [
            plan.evaluate(
                params[g], given[g], outputs,
                extras[g] if extras is not None else None)
            for g, plan in enumerate(self._plans)]
        if extras is not None and self._depth is not None:
            warn_resolution(
                LineResolution(*np.max(resolutions, axis=0)),
                self._spectral.coords.step[0], 1, self._depth.step)


class TransferScenePlan(ScenePlan):
    """
    The evaluation of a scene by radiative transfer along each line of
    sight (see Transfer), unit by unit, front to back, in the order of the
    scene (see DepthUnit): a galaxy on a depth axis of its own (for its
    largest disk), or a group of galaxies on one axis, each at its
    position (see TransferPlan), from the opacity of the units in front of
    it. One set of cubes of opacity holds the dust of all of them (see
    opacity_cubes). (The light of galaxies without opacity just adds up,
    in any order.)

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
        members = [
            galaxy.transfer_member(
                emitters, galaxy_cubes, scene._line_of_sight.depth_axis(
                    spatial, _bounding_radius(galaxy, emitters)))
            for galaxy, emitters, galaxy_cubes in zip(
                scene._galaxies, scene._emitters, cubes)]
        self._plans: tuple[TransferPlan, ...] = tuple(
            TransferPlan(
                [members[g] for g in unit.galaxies], driver, spatial,
                self._spectral, dtype, scene._lines, scene._band,
                scene._data_unit, slab, subchannels, smoothing)
            for unit in scene._order)
        self._subchannels = subchannels
        # (that of every galaxy)
        self._depth_step = members[0].depth.step
        # The state of the march from one unit to the next: the optical
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
            group_params: ParamValues,
            params: list[ParamValues],
            given: Sequence[dict[int, ParamValues]],
            outputs: dict[str, DeviceArray],
            extras: list[Extras] | None,
            group_extras: Extras | None
    ) -> None:
        for state in (self._opacity_front, self._tau):
            if state is not None:
                self._driver.mem_fill(state, 0)
        # Each unit marches from the dust and the line opacity in front of
        # it, if any; with extras, its march tells how finely it resolves
        # its lines, and the worst gets a warning. The extras of the march
        # of a galaxy are its own; those of a group, the group's.
        dust = line_opacity = False
        resolutions = []
        for unit, plan in zip(self._scene._order, self._plans):
            dust = dust or plan.has_dust()
            line_opacity = line_opacity or plan.has_line_opacity()
            unit_extras = march_extra = None
            if extras is not None:
                unit_extras = [extras[g] for g in unit.galaxies]
                march_extra = unit_extras[0] if unit.name is None \
                    else group_extras.scoped(group=unit.name)
            positions = [
                0.0 if name is None else float(group_params[name])
                for name in unit.separations]
            resolution = plan.evaluate(
                [params[g] for g in unit.galaxies],
                [given[g] for g in unit.galaxies], outputs, unit_extras,
                march_extra, self._opacity_front if dust else None,
                self._tau if line_opacity else None, positions)
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
    that holds all the density of the components that an observation
    evaluates (of the given emitters, and the components with dust), at
    any inclination (see Component.bounding_radius); 0 without extent.
    """
    radii = [
        cmp.bounding_radius()
        for i, cmp in enumerate(galaxy.components())
        if i in emitters or isinstance(cmp.opacity(), DustOpacity)]
    return max(radii, default=0.0)


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
    Scene), which sees the emission of the given type (see
    Observable.emission), in the given band if known (an image), and of
    the unit of its data, if any (see Observable.data_unit). The names of
    the components it sees (all those with that emission, in its band, if
    None) are qualified by the names of their galaxies
    ('galaxy.component') when it sees several.

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
        "a spectral cube, which sees the components that emit lines"
    emitters = _select_emitters(galaxies, components, emission, band, kind)
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
                line_of_sight.check(component)
            except ConfigError as e:
                raise ConfigError(
                    f"component {_label(galaxy, i)}: {e}") from e
    return Scene(
        galaxies, seen.mappings, seen.labels, [tuple(e) for e in emitters],
        lines, band, line_of_sight, order, data_unit)


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
