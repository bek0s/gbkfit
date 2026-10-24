"""
What the components on radial nodes share: the disks (on rings) and the
cones (on shells). The traits that a type of them takes (TraitTable);
the traits on their nodes and the parameters of the traits
(NodalTraits); the plans of their evaluation (NodalPlan), with
their parameter values in one buffer (ParamBuffer), their traits in the
native kernels (native_traits) and the maps of their extra outputs
(ExtraMaps, evaluate_native); and the plan of a component made of one
(NodalComponentPlan).
"""

import abc
import logging
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import astropy.units
import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.math import interpolation
from gbkfit.math.interpolation import Interpolator
from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import gridutils, iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from ._detail import Fluxes, parse_nodes, systemic_velocity
from ._units import DataUnit, OpacityScales, band_factors, line_factors
from .base import ComponentPlan, FieldsLayout, NativeGrid
from .lines import Band, Bands, LinePlacement, Lines
from .roles import BandEmission, Emission, LineEmission
from .traits import (
    DPT_DIRECTION_DEFAULT, DPTrait, HTrait, PressureSupport, Trait, VPTrait,
    trait_desc)


_log = logging.getLogger(__name__)


# The most traits of one kind that the native kernels take (TRAIT_NUM_MAX
# in constants.hpp)
MAX_TRAITS = 16


@dataclass(frozen=True)
class TraitTable:
    """
    The traits that a type of component takes: the classes of the traits
    of each of its kinds (e.g. 'vpt'), the directions of its velocity
    polar traits (see traits.VPT_DIRECTIONS; 'plane' for the planar ones,
    see traits.VPTraitPlanar) and those of its dispersion polar traits
    (see traits.DPT_DIRECTIONS).
    """
    component: str
    classes: dict[str, tuple[type[Trait], ...]]
    directions: tuple[str, ...]
    dispersion_directions: tuple[str, ...]

    def check(self, traits_: dict[str, Sequence[Trait]]) -> None:
        """
        Raise ConfigError if the component does not take one of the given
        traits (by kind).
        """
        for kind, values in traits_.items():
            classes = self.classes[kind]
            for trait in values:
                desc = trait_desc(trait.__class__)
                if trait.__class__ not in classes:
                    raise ConfigError(
                        f"a {self.component} does not take {desc}; it takes "
                        f"the types {[cls.type() for cls in classes]}")
                if isinstance(trait, VPTrait) \
                        and trait.direction() not in self.directions:
                    raise ConfigError(
                        f"a {self.component} takes velocity traits along "
                        f"{list(self.directions)}; {desc} is along "
                        f"{trait.direction()!r}")
                if isinstance(trait, DPTrait) and trait.direction() \
                        not in self.dispersion_directions:
                    raise ConfigError(
                        f"a {self.component} takes dispersion traits along "
                        f"{list(self.dispersion_directions)}; {desc} is "
                        f"along {trait.direction()!r}")


def with_default_height_traits(
        value: tuple[HTrait | None, ...] | None,
        npolar: int,
        default: type[HTrait]
) -> tuple[HTrait, ...]:
    """
    Return height traits, one for each of npolar polar traits, with the
    missing ones (None, or all if value is None) of the default type.
    """
    if value is None:
        value = (None,) * npolar
    return tuple(default() if trait is None else trait for trait in value)


@dataclass
class TraitParams:
    """
    The parameters of the traits of one kind, with prefixed names, and for
    each one where its values are given if it is node-wise (its trait's
    sampling, see traits.SAMPLINGS) or None, and its name in its trait.
    The keys of all dicts are in the same order: the order of the traits,
    and for each trait, the smooth parameters before the node-wise ones.
    circular_velocity has the names of those whose values are the
    circular velocity of the mass of the galaxy (see
    traits.Trait.circular_velocity_params), pressure_support those of
    them corrected for the pressure of the disk, with how (see
    traits.Trait.pressure_support), rotation_factor those whose
    values are the factors of the rotation in the gravity at a height,
    with their heights (see traits.Trait.rotation_factor_params), wave
    those whose values are the tables of a density wave in the rotation
    of the disk (see traits.Trait.wave_params), and image those whose
    values are the face-on light of an image (see
    traits.Trait.image_params).
    """
    pdescs: dict[str, ParamDesc]
    sampling: dict[str, str | None]
    pnames: tuple[dict[str, str], ...]
    circular_velocity: tuple[str, ...]
    pressure_support: dict[str, PressureSupport]
    rotation_factor: dict[str, float]
    wave: tuple[str, ...]
    image: tuple[str, ...]


def _trait_params(
        traits_: Sequence[Trait],
        prefix: str | Sequence[str] | None,
        nrnodes: int,
        nsubrnodes: int
) -> TraitParams:
    """
    Return the parameters of the traits of one kind, of the given prefix
    (followed by the index of each trait after the first), or of the given
    prefix of each trait. A node-wise parameter has a value for each node,
    or for each subnode if its trait samples it at the subnodes.
    """
    params_list = []
    for trait in traits_:
        sampling = trait.sampling()
        nnodes = nsubrnodes if sampling == 'subrings' else nrnodes
        params_list.append(
            {name: (pdesc, None)
             for name, pdesc in trait.params_sm().items()}
            | {name: (pdesc, sampling)
               for name, pdesc in trait.params_rnw(nnodes).items()})
    prefixes = list(prefix) if iterutils.is_sequence(prefix) \
        else parseutils.item_prefixes(
            [None] * len(params_list), 'traits', prefix, True)
    params, mappings = iterutils.merge_with_prefixes(params_list, prefixes)
    return TraitParams(
        pdescs={name: tuple_[0] for name, tuple_ in params.items()},
        sampling={name: tuple_[1] for name, tuple_ in params.items()},
        pnames=mappings,
        circular_velocity=tuple(
            mapping[name]
            for trait, mapping in zip(traits_, mappings)
            for name in trait.circular_velocity_params()),
        pressure_support={
            mapping[name]: trait.pressure_support()
            for trait, mapping in zip(traits_, mappings)
            if trait.pressure_support() is not None
            for name in trait.circular_velocity_params()},
        rotation_factor={
            mapping[name]: height
            for trait, mapping in zip(traits_, mappings)
            for name, height in trait.rotation_factor_params().items()},
        wave=tuple(
            mapping[name]
            for trait, mapping in zip(traits_, mappings)
            for name in trait.wave_params()),
        image=tuple(
            mapping[name]
            for trait, mapping in zip(traits_, mappings)
            for name in trait.image_params()))


def _trait_constants(
        traits_: Sequence[Trait], nnodes: int, nsubnodes: int
) -> tuple[list[int], list[float], list[int], list[int]]:
    """
    Return the uids, constant values and their counts, and the parameter
    value counts of a set of traits. Each node-wise parameter has one
    value for each subnode, because the node values are interpolated.
    """
    uids, cvalues, ccounts, pcounts = [], [], [], []
    for trait in traits_:
        consts = trait.consts()
        npvalues_sm = sum(p.size() for p in trait.params_sm().values())
        npvalues_nw = len(trait.params_rnw(nnodes)) * nsubnodes
        uids.append(trait.uid())
        cvalues += consts
        ccounts.append(len(consts))
        pcounts.append(npvalues_sm + npvalues_nw)
    return uids, cvalues, ccounts, pcounts


def fill_param_values(
        values: np.ndarray,
        params: ParamValues,
        pdescs: dict[str, ParamDesc],
        sampling: dict[str, str | None],
        nodes: Sequence[float],
        subnodes: Sequence[float],
        interp: type[Interpolator]
) -> None:
    """
    Write the values of the given parameters one after the other into
    values. The values of the node-wise parameters given at the nodes are
    replaced in params with their values interpolated to the subnodes
    (sampling has where the values of each parameter are given, see
    TraitParams).
    """
    start = 0
    for name in pdescs:
        if sampling[name] == 'rnodes':
            params[name] = interp(nodes, params[name])(subnodes)
        stop = start + np.size(params[name])
        values[start:stop] = params[name]
        start = stop


def subring_radii(
        rnodes: Sequence[float], rstep: float) -> tuple[float, ...]:
    """
    Return the radii of the rings (or shells) that a component of the
    given radial nodes is evaluated on, its subnodes: the first and the
    last node, and between them the centres of rings of equal width, at
    most rstep, that cover the radii between the two exactly.
    """
    nrings = max(1, int(np.ceil((rnodes[-1] - rnodes[0]) / rstep - 1e-9)))
    width = (rnodes[-1] - rnodes[0]) / nrings
    centres = rnodes[0] + (np.arange(nrings) + 0.5) * width
    subrnodes = np.concatenate(([rnodes[0]], centres, [rnodes[-1]]))
    return tuple(typing.cast(list, subrnodes.tolist()))


def _weighted_mean(weighted_sum: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    Return the mean of each voxel from its weighted sum; NaN without
    weight.
    """
    return np.divide(
        weighted_sum, weight, out=np.full_like(weighted_sum, np.nan),
        where=weight != 0)


class NodalTraits:
    """
    The traits of a component on radial nodes, by kind, and their
    parameters (see TraitParams): named with the prefix of their kind in
    prefixes (e.g. 'pt' for the density polar traits), or with that of
    each trait (e.g. 'op_halpha_'), and those node-wise with a value for
    each node, or for each subnode where their trait samples them. A kind
    without traits has no parameters.
    """

    def __init__(
            self,
            kinds: Sequence[str],
            traits_: dict[str, Sequence[Trait]],
            prefixes: dict[str, str | Sequence[str]],
            rnodes: Sequence[float],
            subrnodes: Sequence[float],
            interp: type[Interpolator]
    ):
        self._rnodes = rnodes
        self._subrnodes = subrnodes
        self._interp = interp
        self._traits = {kind: traits_.get(kind, ()) for kind in kinds}
        self._params = {
            kind: _trait_params(
                traits_of_kind, prefixes.get(kind), len(rnodes),
                len(subrnodes))
            for kind, traits_of_kind in self._traits.items()}
        # The parameters whose values are the circular velocity of the
        # mass of their galaxy, and the radii of their values
        self._circular_velocity_params = {
            name: subrnodes if params.sampling[name] == 'subrings' else rnodes
            for params in self._params.values()
            for name in params.circular_velocity}
        self._pressure_supported_params = {
            name: mode for params in self._params.values()
            for name, mode in params.pressure_support.items()}
        self._rotation_factor_params = {
            name: height for params in self._params.values()
            for name, height in params.rotation_factor.items()}
        self._wave_params = tuple(
            name for params in self._params.values() for name in params.wave)
        self._image_params = tuple(
            name for params in self._params.values()
            for name in params.image)

    def kinds(self) -> tuple[str, ...]:
        """Return the kinds of its traits."""
        return tuple(self._traits)

    def traits(self, kind: str) -> tuple[Trait, ...]:
        """Return the traits of one kind."""
        return self._traits[kind]

    def params(self, kind: str) -> TraitParams:
        """Return the parameters of the traits of one kind."""
        return self._params[kind]

    def pdescs(self) -> dict[str, ParamDesc]:
        """
        Return the parameters of its traits, but those that its galaxy
        gives (see circular_velocity_params, rotation_factor_params,
        wave_params and image_params).
        """
        pdescs = {}
        for params in self._params.values():
            pdescs.update(params.pdescs)
        for name in [*self._circular_velocity_params,
                     *self._rotation_factor_params, *self._wave_params,
                     *self._image_params]:
            del pdescs[name]
        return pdescs

    def circular_velocity_params(self) -> dict[str, tuple[float, ...]]:
        """
        Return the parameters whose values are not given, but are the
        circular velocity of the mass of their galaxy at the given radii
        (arcsec), by name.
        """
        return self._circular_velocity_params

    def pressure_supported_params(self) -> dict[str, PressureSupport]:
        """
        Return how those of its circular velocity parameters whose values
        are corrected for the pressure of their component are corrected,
        by their names (see traits.VPTraitMass).
        """
        return self._pressure_supported_params

    def rotation_factor_params(self) -> dict[str, float]:
        """
        Return the parameters whose values are not given, but are the
        factors of the rotation in the gravity of their galaxy at a
        height, at the subnodes: their heights (arcsec), by name (see
        traits.VHTraitMass).
        """
        return self._rotation_factor_params

    def wave_params(self) -> tuple[str, ...]:
        """
        Return the names of the parameters whose values are not given,
        but are the tables of a density wave in the rotation of their
        disk, which its galaxy computes, at the subnodes (see
        traits.VPTraitSpiral).
        """
        return self._wave_params

    def image_params(self) -> tuple[str, ...]:
        """
        Return the names of the parameters whose values are not given,
        but are the face-on light of an image, which its galaxy makes
        (see traits.RPTraitImage).
        """
        return self._image_params

    def values(
            self, kind: str, params: ParamValues
    ) -> tuple[dict[str, np.ndarray], ...]:
        """
        Return the values of the parameters of each trait of one kind, by
        their names in the trait, from the values of the parameters of its
        component: the node-wise ones at the subnodes, as the component is
        evaluated.
        """
        trait_params = self._params[kind]
        out = []
        for names in trait_params.pnames:
            values = {}
            for name, full in names.items():
                value = np.asarray(params[full], dtype=float)
                if trait_params.sampling[full] == 'rnodes':
                    value = self._interp(self._rnodes, value)(self._subrnodes)
                values[name] = value
            out.append(values)
        return tuple(out)

    def constants(
            self, kind: str
    ) -> tuple[list[int], list[float], list[int], list[int]]:
        """
        Return the uids, constant values and their counts, and the
        parameter value counts of the traits of one kind, as the native
        kernels take them: with each node-wise parameter at the subnodes.
        """
        return _trait_constants(
            self._traits[kind], len(self._rnodes), len(self._subrnodes))

    def sizes(self) -> dict[str, int]:
        """
        Return the number of parameter values of the traits of each kind,
        as the native kernels take them (see constants).
        """
        return {kind: sum(self.constants(kind)[3]) for kind in self._traits}

    def fill(self, buffer: 'ParamBuffer', params: ParamValues) -> None:
        """
        Write the parameter values of the traits of each kind into its
        group of the buffer, with the node-wise ones at the subnodes:
        those given at the nodes are interpolated, and replaced in params
        (see fill_param_values).
        """
        for kind, trait_params in self._params.items():
            fill_param_values(
                buffer.host(kind), params, trait_params.pdescs,
                trait_params.sampling, self._rnodes, self._subrnodes,
                self._interp)


class ParamBuffer:
    """
    Values in named groups of the given sizes, packed in one host buffer
    and its device copy, so that they are copied to the device at once.
    """

    def __init__(self, driver: Driver, dtype: np.dtype, sizes: dict[str, int]):
        self._driver = driver
        self._values = driver.mem_alloc_s(sum(sizes.values()), dtype)
        self._host, self._device = {}, {}
        start = 0
        for name, size in sizes.items():
            self._host[name] = self._values[0][start:start + size]
            self._device[name] = self._values[1][start:start + size]
            start += size

    def host(self, name: str) -> np.ndarray:
        """Return the host memory of a group."""
        return self._host[name]

    def device(self, name: str) -> DeviceArray:
        """Return the device memory of a group."""
        return self._device[name]

    def copy_to_device(self) -> None:
        """Copy all the values to the device."""
        self._driver.mem_copy_h2d(*self._values)


def native_traits(
        driver: Driver,
        dtype: np.dtype,
        traits: NodalTraits,
        buffer: ParamBuffer
) -> dict[str, Any]:
    """
    Return the traits of a component as its native class takes them: the
    native set of the traits of each kind that has traits, with the values
    of their parameters in the group of their kind in the buffer, and
    with velocity polar traits, the direction of each ('vpt_directions'),
    and with dispersion polar traits not all isotropic, the direction of
    each ('dpt_directions'; without it, the kernels take them all as
    isotropic).
    """

    def to_device(values: list, dtype_: Any) -> DeviceArray:
        return driver.mem_copy_h2d(np.array(values, dtype_))

    trait_set_class = driver.native_class('TraitSet', dtype)
    result = {}
    for kind in traits.kinds():
        if traits.traits(kind):
            uids, cvalues, ccounts, pcounts = traits.constants(kind)
            result[kind] = trait_set_class(
                uids=to_device(uids, np.int32),
                cvalues=to_device(cvalues, dtype),
                ccounts=to_device(ccounts, np.int32),
                pvalues=buffer.device(kind),
                pcounts=to_device(pcounts, np.int32))
    if 'vpt' in result:
        result['vpt_directions'] = to_device(
            [trait.direction_uid() for trait in traits.traits('vpt')],
            np.int32)
    if 'dpt' in result and any(
            trait.direction() != DPT_DIRECTION_DEFAULT
            for trait in traits.traits('dpt')):
        result['dpt_directions'] = to_device(
            [trait.direction_uid() for trait in traits.traits('dpt')],
            np.int32)
    return result


class ExtraMaps:
    """
    The maps of the extra outputs that the plan of a component writes, on
    its 3d grid (each if asked for): the density of the component, the
    means of its velocity and dispersion, weighted by the absolute density
    (the kernels add to their weighted sums and to the sum of the
    weights), and its density after the opacity that dims it.
    """

    def __init__(
            self,
            driver: Driver,
            dtype: np.dtype,
            grid: NativeGrid,
            density: bool,
            velocity: bool,
            dispersion: bool,
            opacity: bool
    ):
        shape = tuple(grid['spat_size'][::-1])

        def zeros() -> DeviceArray:
            array = driver.mem_alloc_d(shape, dtype)
            driver.mem_fill(array, 0)
            return array

        self._driver = driver
        self._density = zeros() if density else None
        self._velocity = zeros() if velocity else None
        self._dispersion = zeros() if dispersion else None
        self._weight = zeros() if velocity or dispersion else None
        self._opacity = zeros() if opacity else None

    def native(self) -> dict[str, DeviceArray | None]:
        """Return the maps as the native evaluation functions take them."""
        return dict(
            rdata_cmp=self._density, ordata_cmp=self._opacity,
            vdata_cmp=self._velocity, ddata_cmp=self._dispersion,
            vdweight_cmp=self._weight)

    def collect(self, out_extra: dict[str, Any], density_key: str) -> None:
        """
        Add the maps to the extra outputs: the density by density_key
        (e.g. 'bdata' for light, 'odata' for dust), the velocity
        ('vdata'), the dispersion ('ddata') and the density after the
        opacity ('obdata').
        """
        driver = self._driver
        if self._density is not None:
            out_extra[density_key] = driver.mem_copy_d2h(self._density)
        if self._weight is not None:
            weight = driver.mem_copy_d2h(self._weight)
        if self._velocity is not None:
            out_extra['vdata'] = _weighted_mean(
                driver.mem_copy_d2h(self._velocity), weight)
        if self._dispersion is not None:
            out_extra['ddata'] = _weighted_mean(
                driver.mem_copy_d2h(self._dispersion), weight)
        if self._opacity is not None:
            out_extra['obdata'] = driver.mem_copy_d2h(self._opacity)
        for key, present in (
                (density_key, self._density is not None),
                ('vdata', self._velocity is not None),
                ('ddata', self._dispersion is not None)):
            if present:
                sumabs = np.nansum(np.abs(out_extra[key]))
                _log.debug(f"sum(abs({key})): {sumabs}")


def evaluate_native(
        evaluate: Callable[[dict[str, Any]], None],
        driver: Driver,
        dtype: np.dtype,
        traits: NodalTraits,
        grid: NativeGrid,
        outputs: dict[str, DeviceArray | None],
        out_extra: dict[str, Any] | None,
        density_key: str,
        lines: DeviceArray | None,
        rdata_scale: float,
        image_scale: float
) -> None:
    """
    Evaluate a component on radial nodes of the given traits (see
    NodalPlan.evaluate): call evaluate with the keyword arguments of its
    native evaluation function (the grid, the outputs, the device memory
    of its lines and the scales), with the maps of its extra outputs if
    they are wanted (see ExtraMaps), which then go to out_extra.
    """
    maps = None
    if out_extra is not None:
        maps = ExtraMaps(
            driver, dtype, grid,
            density=bool(traits.traits('rpt')),
            velocity=bool(traits.traits('vpt')),
            dispersion=bool(traits.traits('dpt')),
            opacity=outputs.get('opacity') is not None)
    evaluate(
        grid | outputs | (maps.native() if maps is not None else {})
        | dict(lines=lines, rdata_scale=rdata_scale, image_scale=image_scale))
    if maps is not None:
        maps.collect(out_extra, density_key)


class NodalPlan(abc.ABC):
    """
    The evaluation of a component on radial nodes on a driver and dtype,
    which owns the memory it needs: what the plan of its component needs
    of it (see NodalComponentPlan).
    """

    @abc.abstractmethod
    def evaluate(
            self,
            params: ParamValues,
            grid: NativeGrid,
            outputs: dict[str, DeviceArray | None],
            out_extra: dict[str, Any] | None,
            density_key: str,
            lines: np.ndarray | None = None,
            rdata_scale: float = 1.0,
            image_scale: float = 1.0
    ) -> None:
        """
        Add the component to the outputs. grid has the grid of the native
        evaluation functions (see NativeGrid), and outputs the arrays they
        add to (all optional): the cubes of opacity they read ('opacity')
        with their ratios ('opacity_ratios'), the 'image' (whose light is
        times image_scale) or 'scube', the 3d density ('rdata', times
        rdata_scale) and density after the opacity ('ordata'), or instead
        of the light, the fields of the march, which they write ('fields',
        see fields_layout). density_key names the density map in the extra
        outputs (e.g. 'bdata' for light, 'odata' for dust; see ExtraMaps).
        lines has the offset, scale and flux of each emission line (an
        array of shape (nlines, 3); see the native GridArgs), if the plan
        has lines.
        """
        pass

    @abc.abstractmethod
    def fields_layout(self) -> FieldsLayout:
        """
        Return how the component holds its fields for the march of
        radiative transfer (see the native FieldsArgs::fields).
        """
        pass


def parse_component_rnode_args(
        rnmin: float | None,
        rnmax: float | None,
        rnsep: float | None,
        rnlen: int | None,
        rnodes: Any,
        rstep: float | None,
        interp: str
) -> dict[str, Any]:
    """
    Return the radial nodes of a component (the rings of a disk, the
    shells of a cone; see parse_nodes), the step of its subnodes (by
    default half the smallest separation of the nodes, at most 1), and the
    interpolation class of its node-wise parameters. Raise ConfigError for
    invalid options.
    """
    nodes = parse_nodes('r', rnmin, rnmax, rnsep, rnlen, rnodes)
    step = rstep
    if step is None:
        # (a float, as configurations take it)
        step = float(min(1, min(np.diff(nodes)) / 2))
    # (half the difference is allowed, also as it rounds)
    if step <= 0 or step > min(np.diff(nodes)) / 2 * (1 + 1e-9):
        raise ConfigError(
            "rstep must be greater than zero and less than half the "
            "smallest difference between two consecutive rnodes")
    interpolations = dict(
        linear=interpolation.InterpolatorLinear,
        akima=interpolation.InterpolatorAkima,
        pchip=interpolation.InterpolatorPCHIP)
    if interp not in interpolations:
        raise ConfigError(
            "interp must be one of the following: "
            f"{list(interpolations.keys())}")
    return dict(rnodes=nodes, rstep=step, interp=interpolations[interp])


def make_lines(
        pdescs: dict[str, ParamDesc], emission: Emission | None
) -> Lines | None:
    """
    Return the emission lines of a component (see Lines), if it emits
    lines, whose parameters must have other names than its parameters
    (pdescs).
    """
    if not isinstance(emission, LineEmission):
        return None
    lines = emission.lines()
    if repeated := sorted(set(lines.pdescs()) & set(pdescs)):
        raise ConfigError(
            f"the parameters of the lines have the names of other "
            f"parameters: {repeated}; rename the lines")
    return lines


def make_bands(
        pdescs: dict[str, ParamDesc], emission: Emission | None
) -> Bands | None:
    """
    Return the bands of a component (see Bands), if it emits in bands,
    whose parameters must have other names than its parameters (pdescs).
    """
    if not isinstance(emission, BandEmission):
        return None
    bands = emission.bands()
    if repeated := sorted(set(bands.pdescs()) & set(pdescs)):
        raise ConfigError(
            f"the parameters of the bands have the names of other "
            f"parameters: {repeated}; rename the bands")
    return bands


@dataclass(frozen=True)
class EmissionSetup:
    """
    What the plan of a component on radial nodes evaluates of its light
    (see emission_setup): the number of lines its plan adds to spectral
    cubes (0 in an image, or without light), the row of the line of each
    of its traits of line opacity among them (-1 for a line it does not
    evaluate; None without lines), and for the plan of its component (see
    NodalComponentPlan), the fluxes of its lines or band, the place of
    each line on the spectral axis, and the scales of its line opacity.
    """
    nlines: int = 0
    opacity_rows: list[int] | None = None
    fluxes: Fluxes | None = None
    lines: LinePlacement | None = None
    opacity_scales: OpacityScales | None = None


def emission_setup(
        lines: Lines | None,
        bands: Bands | None,
        unit: astropy.units.UnitBase | None,
        spectral: gridutils.Grid,
        names: tuple[str, ...] | None,
        band: Band | None,
        emits: bool,
        data_unit: DataUnit | None,
        redshift: float,
        opacity_lines: Sequence[int]
) -> EmissionSetup:
    """
    Return what the plan of a component evaluates of its light (see
    EmissionSetup), with the given emission lines or bands (if any) and
    the unit of its light (if any): if it emits, the lines of the given
    names (all if None), at the redshift of its galaxy, or the given band
    (if known), in the unit of the data (see Component.plan). opacity_lines
    has the line of each of its traits of line opacity: its index among
    its lines.
    """
    if not emits:
        return EmissionSetup()
    if lines is None:
        ratio = bands.ratio_name(band)
        return EmissionSetup(fluxes=Fluxes(
            1, [(0, ratio)] if ratio is not None else [],
            band_factors(unit, data_unit, band, redshift)))
    # (and the row of the line of each trait of line opacity among the
    # lines it evaluates)
    selected = lines.select(names)
    opacity_rows = [
        selected.index(line) if line in selected else -1
        for line in opacity_lines]
    factors = lines.factors(spectral, redshift)[list(selected)]
    # (with a unit, the frequency at which the galaxy emits each line)
    frequencies = None if unit is None \
        else lines.frequencies(spectral, redshift)[list(selected)]
    ratios = [
        (row, lines.ratio_name(index)) for row, index in enumerate(selected)
        if lines.ratio_name(index) is not None]
    # (with a unit, the coefficients of line opacity are 1/S in 1/K)
    opacity_scales = None
    if opacity_lines and unit is not None:
        opacity_scales = OpacityScales(
            unit, data_unit, factors, spectral, frequencies)
    return EmissionSetup(
        len(selected), opacity_rows,
        Fluxes(len(factors), ratios, line_factors(
            unit, data_unit, factors, spectral, frequencies)),
        LinePlacement(factors, spectral), opacity_scales)


class NodalComponentPlan(ComponentPlan):
    """
    The evaluation of a component made of one disk or cone: the plan of
    the disk or cone (see NodalPlan), which adds its light to the outputs,
    dimmed by the opacity with the given ratios (see Component.plan), or
    its optical depth to a cube of opacity. Its light is that of its lines
    (where each is on the spectral axis at its systemic velocity), or in
    an image, that of its band, of their fluxes (see EmissionSetup). With
    line opacity and a unit, the fields of the opacity of each line are
    times its scale to brightness temperature, as its coefficients are
    1/S in 1/K.
    """

    def __init__(
            self,
            driver: Driver,
            plan: NodalPlan,
            emission: EmissionSetup,
            opacity_ratios: np.ndarray | None = None
    ):
        self._driver = driver
        self._plan = plan
        self._fluxes = emission.fluxes
        self._opacity_scales = emission.opacity_scales
        self._opacity_ratios = None
        if opacity_ratios is not None:
            self._opacity_ratios = driver.mem_copy_h2d(opacity_ratios)
        # The offset, scale and flux of each line (set at each evaluation)
        self._lines = emission.lines
        self._rows = None if self._lines is None \
            else np.zeros((len(self._lines), 3))

    def line_values(self, params: ParamValues) -> np.ndarray:
        if self._lines is None:
            return np.array([[0.0, 0.0, self._image_scale(params)]])
        self._rows[:, :2] = self._lines(systemic_velocity(params))
        self._rows[:, 2] = self._fluxes(params)
        return self._rows

    def evaluate_emission(
            self,
            params: ParamValues,
            grid: NativeGrid,
            outputs: dict[str, DeviceArray | None],
            out_extra: dict[str, Any] | None
    ) -> None:
        lines = self.line_values(params) if self._lines is not None \
            else None
        odata = outputs.get('odata')
        plan_outputs = dict(
            image=outputs.get('image'),
            scube=outputs.get('scube'),
            rdata=outputs.get('bdata'),
            opacity=odata,
            opacity_ratios=self._opacity_ratios if odata is not None
            else None,
            ordata=outputs.get('obdata'))
        self._plan.evaluate(
            params, grid, plan_outputs, out_extra, 'bdata', lines,
            image_scale=self._image_scale(params))

    def fields_layout(self) -> FieldsLayout:
        return self._plan.fields_layout()

    def evaluate_fields(
            self,
            params: ParamValues,
            grid: NativeGrid,
            fields: DeviceArray,
            out_extra: dict[str, Any] | None
    ) -> None:
        self._driver.mem_fill(fields, 0)
        self._plan.evaluate(
            params, grid, dict(fields=fields), out_extra, 'bdata')
        if self._opacity_scales is not None:
            # (the fields of line opacity follow those of the light, its
            # velocity and its dispersion: one for each line)
            scales = self._opacity_scales(systemic_velocity(params))
            for row, scale in enumerate(scales):
                slot = fields[3 + row]
                self._driver.math_mul(slot, slot.dtype.type(scale), out=slot)

    def evaluate_opacity(
            self,
            params: ParamValues,
            grid: NativeGrid,
            odata: DeviceArray,
            scale: float,
            out_extra: dict[str, Any] | None
    ) -> None:
        # The density of the component is its optical depth
        self._plan.evaluate(
            params, grid, dict(rdata=odata), out_extra, 'odata',
            rdata_scale=scale)

    def _image_scale(self, params: ParamValues) -> float:
        """
        Return the flux of the light in the band of an image (see Fluxes);
        1 in a spectral cube, whose lines have theirs.
        """
        if self._lines is not None:
            return 1.0
        return float(self._fluxes(params)[0])
