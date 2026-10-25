"""
Scenes: galaxies drawn together on one grid (e.g. those an observation
sees; see gbkfit.observation), and their evaluation on the grid of an
image or of a spectral cube: their light added up (AdditiveScenePlan), or
marched through, front to back, by radiative transfer
(TransferScenePlan).
"""

import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamValues
from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import ConfigError
from ..extras import Extras
from ._component_plans import opacity_cubes
from ._component_set import IMAGE_SPECTRAL_AXIS, ComponentSetPlan
from ._stage import GalaxyState
from ._transfer import (
    MAX_SUBCHANNELS, LineResolution, TransferPlan, warn_resolution)
from ._units import DataUnit
from .base import DepthAxis, FrameParams
from .lines import Band
from .roles import DustOpacity
from .traits import RPTraitImage

if TYPE_CHECKING:
    from ..galaxy import Galaxy


__all__ = [
    'MAX_SUBCHANNELS',
    'AdditiveScenePlan',
    'DepthUnit',
    'Scene',
    'ScenePlan',
    'TransferScenePlan',
    'named_components'
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


class Scene:
    """
    Galaxies drawn together on one grid (e.g. those an observation sees;
    see gbkfit.observation): the galaxies, the indices of the components
    of each whose light is drawn (its emitters; the opacity of all
    components applies), the emission lines drawn (all if None) or the
    band (if known), the order of the galaxies along the line of sight,
    front to back (in units: galaxies or groups of them; those it leaves
    out after them, in their order), and the unit of the data, which the
    light of the emitters takes (see Component.plan; None without data).
    The parameters of each galaxy have the names of the mapping of the
    galaxy among those of its group of galaxies (e.g. ObservationGroup),
    and its extra outputs its label as the id 'galaxy' (those of a group
    of them, its name as the id 'group').
    """

    def __init__(
            self,
            galaxies: Sequence['Galaxy'],
            mappings: Sequence[dict[str, str]],
            labels: Sequence[str],
            emitters: Sequence[tuple[int, ...]],
            lines: tuple[str, ...] | None,
            band: Band | None,
            order: Sequence[DepthUnit] | None = None,
            data_unit: DataUnit | None = None
    ):
        self._galaxies = tuple(galaxies)
        self._mappings = tuple(mappings)
        self._labels = tuple(labels)
        self._emitters = tuple(emitters)
        self._lines = lines
        self._band = band
        self._data_unit = data_unit
        order = tuple(order or ())
        ordered = {g for unit in order for g in unit.galaxies}
        self._order = order + tuple(
            DepthUnit((g,)) for g in range(len(self._galaxies))
            if g not in ordered)

    def bounding_radii(self) -> tuple[float, ...]:
        """
        Return the bounding radius (arcsec) of each galaxy: that of its
        largest component that is drawn (see _bounding_radius); 0 without
        extent.
        """
        return tuple(
            _bounding_radius(galaxy, emitters)
            for galaxy, emitters in zip(self._galaxies, self._emitters))

    def image_traits(self) -> tuple[RPTraitImage, ...]:
        """
        Return the image traits of the disks whose light is drawn (see
        traits.RPTraitImage).
        """
        return tuple(
            trait
            for galaxy, emitters in zip(self._galaxies, self._emitters)
            for i in emitters
            if (disk := galaxy.components()[i].density_disk()) is not None
            for trait in disk.traits('rpt')
            if isinstance(trait, RPTraitImage))

    def frame_params(self, name: str) -> FrameParams:
        """
        Return the parameters that place the component of the given name
        (see named_components) in a frame (see Component.frame_params), by
        their names among those of the group of galaxies. Raise ConfigError
        if no component has the name.
        """
        g, i = named_components(self._galaxies, [name])[name]
        galaxy = self._galaxies[g]
        names = galaxy.component_params(i)
        return FrameParams(*(
            None if own is None else self._mappings[g][names[own]]
            for own in galaxy.components()[i].frame_params()))


class ScenePlan(abc.ABC):
    """
    The evaluation of a scene on a grid (e.g. along the line of sight of
    an observation; see gbkfit.observation.LineOfSight.plan_scene): the
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
    by its own opacity only (e.g. along the lines of sight Flat and Grid
    of observations): on the given depth axis, if any (for the largest
    disk of the scene). With extra outputs, it logs a warning if the depth
    axis resolves the velocity of the light too coarsely (see
    warn_resolution).
    """

    def __init__(
            self,
            scene: Scene,
            driver: Driver,
            grid: gridutils.Grid,
            depth: DepthAxis | None,
            dtype: np.dtype,
            smoothing: float | None
    ):
        super().__init__(scene, grid)
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
    sight (e.g. the line of sight Transfer of observations), unit by unit,
    front to back, in the order of the scene (see DepthUnit): a galaxy on
    its depth axis (the given one of each galaxy, for its largest disk),
    or a group of galaxies on one axis, each at its position (see
    TransferPlan), from the opacity of the units in front of it. One set
    of cubes of opacity holds the dust of all of them (see opacity_cubes).
    (The light of galaxies without opacity just adds up, in any order.)

    With extra outputs, it logs a warning if the march resolves line
    opacity, or the depth axis the velocity of the light, too coarsely
    (see warn_resolution).
    """

    def __init__(
            self,
            scene: Scene,
            driver: Driver,
            grid: gridutils.Grid,
            depths: Sequence[DepthAxis],
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
            galaxy.transfer_member(emitters, galaxy_cubes, depth)
            for galaxy, emitters, galaxy_cubes, depth in zip(
                scene._galaxies, scene._emitters, cubes, depths)]
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


def _bounding_radius(galaxy: 'Galaxy', emitters: Sequence[int]) -> float:
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


def named_components(
        galaxies: Sequence['Galaxy'], names: Sequence[str]
) -> dict[str, tuple[int, int]]:
    """
    Return the indices of the galaxy and of the component of each named
    component of galaxies drawn together, by its name ('galaxy.component'
    when there are several galaxies).

    Raises
    ------
    ConfigError
        If a given name is not among them.
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
