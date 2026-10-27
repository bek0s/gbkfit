"""
What the components evaluated by native kernels share (the disks, the
cones and the spheroids): the plans of their evaluation (NativePlan),
with their parameter values in one buffer (ParamBuffer) and the maps of
their extra outputs (ExtraMaps, evaluate_native); what they evaluate of
their light (make_lines, make_bands, emission_setup); and the plan of a
component made of one (NativeComponentPlan).
"""

import abc
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import astropy.units
import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import ConfigError
from ._detail import Fluxes, systemic_velocity
from ._units import DataUnit, OpacityScales, band_factors, line_factors
from .base import ComponentPlan, ComponentState, FieldsLayout, NativeGrid
from .lines import Band, Bands, LinePlacement, Lines
from .roles import STARS, BandEmission, Emission, LineEmission


_log = logging.getLogger(__name__)


def _weighted_mean(weighted_sum: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    Return the mean of each voxel from its weighted sum; NaN without
    weight.
    """
    return np.divide(
        weighted_sum, weight, out=np.full_like(weighted_sum, np.nan),
        where=weight != 0)


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
        grid: NativeGrid,
        outputs: dict[str, DeviceArray | None],
        out_extra: dict[str, Any] | None,
        density_key: str,
        lines: DeviceArray | None,
        rdata_scale: float,
        image_scale: float,
        *,
        velocity: bool,
        dispersion: bool
) -> None:
    """
    Evaluate a component natively (see NativePlan.evaluate): call evaluate
    with the keyword arguments of its native evaluation function (the
    grid, the outputs, the device memory of its lines and the scales),
    with the maps of its extra outputs if they are wanted (see ExtraMaps),
    which then go to out_extra: its density, and its velocity and
    dispersion if it has them (velocity and dispersion).
    """
    maps = None
    if out_extra is not None:
        maps = ExtraMaps(
            driver, dtype, grid, density=True, velocity=velocity,
            dispersion=dispersion,
            opacity=outputs.get('opacity') is not None)
    evaluate(
        grid | outputs | (maps.native() if maps is not None else {})
        | dict(lines=lines, rdata_scale=rdata_scale, image_scale=image_scale))
    if maps is not None:
        maps.collect(out_extra, density_key)


class NativePlan(abc.ABC):
    """
    The evaluation of a component by native kernels on a driver and
    dtype, which owns the memory it needs: what the plan of its component
    needs of it (see NativeComponentPlan).
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
            image_scale: float = 1.0,
            state: ComponentState | None = None
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
        has lines, and state the state of the component in the evaluation
        (see ComponentStage.state), if it has one.
        """
        pass

    @abc.abstractmethod
    def fields_layout(self) -> FieldsLayout:
        """
        Return how the component holds its fields for the march of
        radiative transfer (see the native FieldsArgs::fields).
        """
        pass


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
    NativeComponentPlan), the fluxes of its lines or band, the place of
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
        fluxes = Fluxes(
            1, [(0, ratio)] if ratio is not None else [],
            band_factors(unit, data_unit, band, redshift))
        if names is not None and STARS in names:
            # (the light of stars in a spectral cube: one line, without a
            # rest, of the flux of its band; see StarsEmission)
            return EmissionSetup(
                nlines=1, fluxes=fluxes,
                lines=LinePlacement(np.ones(1), spectral))
        return EmissionSetup(fluxes=fluxes)
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


class NativeComponentPlan(ComponentPlan):
    """
    The evaluation of a component made of one disk, cone or spheroid: its
    plan (see NativePlan), which adds its light to the outputs, dimmed by
    the opacity with the given ratios (see Component.plan), or its optical
    depth to a cube of opacity. Its light is that of its lines (where each
    is on the spectral axis at its systemic velocity), or in an image,
    that of its band, of their fluxes (see EmissionSetup). With line
    opacity and a unit, the fields of the opacity of each line are times
    its scale to brightness temperature, as its coefficients are 1/S in
    1/K.
    """

    def __init__(
            self,
            driver: Driver,
            plan: NativePlan,
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
            out_extra: dict[str, Any] | None,
            state: ComponentState | None
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
            image_scale=self._image_scale(params), state=state)

    def fields_layout(self) -> FieldsLayout:
        return self._plan.fields_layout()

    def evaluate_fields(
            self,
            params: ParamValues,
            grid: NativeGrid,
            fields: DeviceArray,
            out_extra: dict[str, Any] | None,
            state: ComponentState | None
    ) -> None:
        self._driver.mem_fill(fields, 0)
        self._plan.evaluate(
            params, grid, dict(fields=fields), out_extra, 'bdata',
            state=state)
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
            out_extra: dict[str, Any] | None,
            state: ComponentState | None
    ) -> None:
        # The density of the component is its optical depth
        self._plan.evaluate(
            params, grid, dict(rdata=odata), out_extra, 'odata',
            rdata_scale=scale, state=state)

    def _image_scale(self, params: ParamValues) -> float:
        """
        Return the flux of the light in the band of an image (see Fluxes);
        1 in a spectral cube, whose lines have theirs.
        """
        if self._lines is not None:
            return 1.0
        return float(self._fluxes(params)[0])
