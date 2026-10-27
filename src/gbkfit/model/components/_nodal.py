"""
What the components on radial nodes share: the disks (on rings) and the
cones (on shells). The traits that a type of them takes (TraitTable);
the traits on their nodes and the parameters of the traits
(NodalTraits), with their values in the buffer of a plan (see
_native.ParamBuffer) and in the native kernels (native_traits); and
their radial nodes (parse_component_rnode_args).
"""

import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import astropy.units
import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.math import interpolation
from gbkfit.math.interpolation import Interpolator
from gbkfit.params import ParamDesc, ParamValues, pdescs_with_units
from gbkfit.utils import iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from ._detail import parse_nodes
from ._native import ParamBuffer
from .traits import (
    DPT_DIRECTION_DEFAULT, DPTrait, GalaxyInput, HTrait, ImageInput,
    PressureSupport, RPTrait, RotationFactorsInput, RotationInput, Trait,
    VPTrait, WaveInput, trait_desc)


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
        traits (by kind), or two of them share a name.
        """
        names = [
            trait.name() for values in traits_.values() for trait in values
            if trait.name() is not None]
        if repeated := sorted({n for n in names if names.count(n) > 1}):
            raise ConfigError(
                f"two traits of a {self.component} cannot share a name; "
                f"repeated: {repeated}")
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
    The others have those whose values the galaxy gives (see
    traits.Trait.from_galaxy): circular_velocity the names of those whose
    values are the circular velocity of the mass of the galaxy (see
    traits.RotationInput), pressure_support those of them corrected for
    the pressure of the disk, with how, rotation_factor those whose
    values are the factors of the rotation in the gravity at a height,
    with their heights (see traits.RotationFactorsInput), wave those whose
    values are the tables of a density wave in the rotation of the disk
    (see traits.WaveInput), and image those whose values are the face-on
    light of an image (see traits.ImageInput).
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
        nsubrnodes: int,
        density_unit: astropy.units.UnitBase | None
) -> TraitParams:
    """
    Return the parameters of the traits of one kind, of the given prefix
    (followed by the index of each trait after the first), or of the given
    prefix of each trait. A node-wise parameter has a value for each node,
    or for each subnode if its trait samples it at the subnodes. The
    amplitudes of density polar traits are in the given unit of their
    density (None if not known).
    """
    params_list = []
    for trait in traits_:
        sampling = trait.sampling()
        nnodes = nsubrnodes if sampling == 'subrings' else nrnodes
        units = dict.fromkeys(trait.amplitudes(), density_unit) \
            if isinstance(trait, RPTrait) else {}
        params_sm = pdescs_with_units(trait.params_sm(), units)
        params_rnw = pdescs_with_units(trait.params_rnw(nnodes), units)
        params_list.append(
            {name: (pdesc, None) for name, pdesc in params_sm.items()}
            | {name: (pdesc, sampling) for name, pdesc in params_rnw.items()})
    prefixes = list(prefix) if iterutils.is_sequence(prefix) \
        else parseutils.item_prefixes(
            [trait.name() for trait in traits_], 'traits', prefix, True)
    params, mappings = iterutils.merge_with_prefixes(params_list, prefixes)
    inputs = [
        (trait.from_galaxy(), mapping)
        for trait, mapping in zip(traits_, mappings)]
    rotations = [
        (given, mapping) for given, mapping in inputs
        if isinstance(given, RotationInput)]

    def names(kind: type[GalaxyInput]) -> tuple[str, ...]:
        return tuple(
            mapping[name] for given, mapping in inputs
            if isinstance(given, kind) for name in given.params)
    return TraitParams(
        pdescs={name: tuple_[0] for name, tuple_ in params.items()},
        sampling={name: tuple_[1] for name, tuple_ in params.items()},
        pnames=mappings,
        circular_velocity=names(RotationInput),
        pressure_support={
            mapping[name]: given.support for given, mapping in rotations
            if given.support is not None for name in given.params},
        rotation_factor={
            mapping[name]: height for given, mapping in inputs
            if isinstance(given, RotationFactorsInput)
            for name, height in zip(given.params, given.heights)},
        wave=names(WaveInput),
        image=names(ImageInput))


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


class NodalTraits:
    """
    The traits of a component on radial nodes, by kind, and their
    parameters (see TraitParams): named with the prefix of their kind in
    prefixes (e.g. 'pt' for the density polar traits), or with that of
    each trait (e.g. 'op_halpha_'), and those node-wise with a value for
    each node, or for each subnode where their trait samples them. A kind
    without traits has no parameters. The amplitudes of the density polar
    traits of a kind (see RPTrait.amplitudes) are in the unit of their
    density in density_units (e.g. of 'rpt', that of the light of a
    component that emits), or have none if it is not known (or not given).
    """

    def __init__(
            self,
            kinds: Sequence[str],
            traits_: dict[str, Sequence[Trait]],
            prefixes: dict[str, str | Sequence[str]],
            density_units: Mapping[str, astropy.units.UnitBase | None],
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
                len(subrnodes), density_units.get(kind))
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

    def fill(self, buffer: ParamBuffer, params: ParamValues) -> None:
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
