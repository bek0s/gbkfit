"""
The evaluation of the components of a galaxy by radiative transfer along
each line of sight (TransferPlan; see the line of sight Transfer).
"""

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamValues
from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import ConfigError
from ._component_plans import ComponentPlans
from .base import DepthAxis, NativeGrid, OpacityCubes
from .lines import Band

if TYPE_CHECKING:
    from ._component_set import ComponentSet


__all__ = [
    'LineResolution',
    'TransferPlan',
    'warn_resolution'
]


_log = logging.getLogger(__name__)


# The most emitters that the march takes (TRANSFER_EMITTER_NUM_MAX in
# constants.hpp)
_MAX_EMITTERS = 16

# The most sub-channels of a channel that the march takes
# (TRANSFER_SUBCHANNEL_NUM_MAX in constants.hpp)
MAX_SUBCHANNELS = 64

# The resolution of line opacity (see LineResolution): checked where the
# optical depth of the lines at a channel reaches _THIN_TAU, with a
# warning above the given ratios (see warn_resolution). In an edge-on
# disk of HI of optically thick lines (a peak of 180 K without opacity,
# 83 K with), the error of the model is about 1% of its peak at these
# ratios or less: it grows as the square of the spectral ratio (1.1% at
# 0.75, 0.3% at 0.5), and steeply with the depth ratio (0.5% at 2, 3% at
# 2.7, 10% at 4).
_THIN_TAU = 0.1
_MAX_SPECTRAL_RATIO = 0.75
_MAX_DEPTH_RATIO = 2.0


class LineResolution(NamedTuple):
    """
    How finely a march resolves the line opacity of a galaxy, where its
    lines are not optically thin (see TransferPlan.evaluate): the largest
    ratio of the width of the channels of the march (the sub-channels) to
    the dispersion of a line (spectral), and of the change of the velocity
    between adjacent voxels along a line of sight to their dispersion
    (depth). Both are discretisations of the model: the march averages
    the optical depth of a voxel over a channel, and over a voxel. Lines
    of zero dispersion, which no resolution resolves, are left out.
    """
    spectral: float
    depth: float


def warn_resolution(
        resolution: LineResolution,
        channel: float,
        subchannels: int,
        depth_step: float
) -> None:
    """
    Log a warning for each way in which a march, of channels of the given
    width and number of sub-channels and voxels of the given depth
    (arcsec), resolves line opacity too coarsely (see LineResolution),
    with the setting that resolves it.
    """
    if resolution.spectral > _MAX_SPECTRAL_RATIO:
        needed = math.ceil(
            subchannels * resolution.spectral / _MAX_SPECTRAL_RATIO)
        _log.warning(
            f"radiative transfer does not resolve line opacity in velocity: "
            f"the channels of the march ({channel:g} in {subchannels} "
            f"sub-channels) are up to {resolution.spectral:.3g} times as "
            f"wide as the dispersion of the lines where they are not thin; "
            f"raise the option subchannels of the line of sight to "
            f"{min(needed, MAX_SUBCHANNELS)}")
    if resolution.depth > _MAX_DEPTH_RATIO:
        _log.warning(
            f"radiative transfer does not resolve line opacity along the "
            f"line of sight: the velocity changes by up to "
            f"{resolution.depth:.3g} dispersions between adjacent voxels "
            f"where the lines are not thin; lower the step of the depth "
            f"axis (the option depth: {{step: ...}} of the line of sight) "
            f"from {depth_step:g} to "
            f"{depth_step * _MAX_DEPTH_RATIO / resolution.depth:.3g} arcsec")


class TransferPlan:
    """
    The evaluation of the components of a galaxy on the grid of an
    observation by radiative transfer along each line of sight: the
    absorbers write the optical depth of each voxel to the cubes of
    opacity, the emitters write the fields of their light (see
    ComponentPlan.evaluate_fields), and a march along each line of sight,
    front to back, adds the light of each channel that reaches the viewer
    to the image or the spectral cube. The march starts from the opacity
    in front of the galaxy, that of the galaxies in front of it, and adds
    its own, for the galaxies behind it (see evaluate).

    It evaluates the depth axis in slabs of planes, one after the other,
    to bound its memory: the absorbers and the volumes (e.g. thick disks)
    on the planes of each slab (the clouds of a Monte Carlo disk are drawn
    again for each, the same ones), the layers (e.g. thin disks, points)
    once. The parameters are those of ComponentPlans, with a depth axis,
    the redshift of the galaxy, the number of planes of a slab (slab; all
    if None), and the number of sub-channels of each channel that the
    march transfers the light of the channel at, if it carries line
    opacity (subchannels; the light of the channel is their mean).

    Raises
    ------
    ConfigError
        If there are more emitters than the march takes.
    """

    def __init__(
            self,
            component_set: 'ComponentSet',
            driver: Driver,
            grid: gridutils.Grid,
            spectral: gridutils.Grid,
            depth: DepthAxis,
            dtype: np.dtype,
            emitters: Sequence[int],
            cubes: OpacityCubes,
            lines: tuple[str, ...] | None,
            band: Band | None,
            redshift: float,
            slab: int | None,
            subchannels: int
    ):
        if len(emitters) > _MAX_EMITTERS:
            raise ConfigError(
                f"radiative transfer takes at most {_MAX_EMITTERS} components "
                f"that emit; there are {len(emitters)}")
        plans = ComponentPlans(
            component_set, driver, grid, spectral, depth, dtype, emitters,
            cubes, lines, band)
        self._plans = plans
        self._subchannels = subchannels
        native_grid = plans.native_grid
        components = component_set.components()
        # The slabs: their first plane and their number of planes
        nz = native_grid['spat_size'][2]
        size = nz if slab is None else min(slab, nz)
        self._slabs = [(z0, min(size, nz - z0)) for z0 in range(0, nz, size)]
        slab_grid = _slab_grid(native_grid, 0, size)
        # The lines of each emitter, in one buffer (which is copied to the
        # device once per evaluation): one row for each of its wavelengths
        # (those of its lines, or its band)
        nrows = [
            len(components[i].emission().wavelengths(lines, band, redshift))
            for i in plans.emitters]
        ends = np.cumsum(nrows)
        self._rows = {
            i: slice(end - n, end)
            for i, n, end in zip(plans.emitters, nrows, ends)}
        self._lines = driver.mem_alloc_s((sum(nrows), 3), dtype)
        # The fields of each emitter: of the whole grid for a layer, of the
        # planes of a slab for a volume (flat, so that the fields of the
        # last slab, which can have fewer planes, are its first values)
        self._layouts = {
            i: plans.plans[i].fields_layout() for i in plans.emitters}
        self._fields = {
            i: driver.mem_alloc_d(math.prod(layout.shape(
                native_grid if layout.layer else slab_grid)), dtype)
            for i, layout in self._layouts.items()}
        self._ratios = {
            i: None if plans.ratios[i] is None
            else driver.mem_copy_h2d(plans.ratios[i])
            for i in plans.emitters}
        # The cubes of opacity of a slab, with absorbers (flat, as the
        # fields)
        self._odata = None
        if plans.absorbers:
            self._odata = driver.mem_alloc_d(
                plans.ncubes * math.prod(slab_grid['spat_size']), dtype)
        # The emitters of the march of slabs of each size: with the line
        # opacity of their fields, without dust, and with dust if the
        # observation has dust (see evaluate); and without opacity, for
        # the light that no opacity dims (an extra output)
        kinds = [(False, True), (False, False)] \
            + ([(True, True)] if plans.ncubes else [])
        self._emitters = {
            (slab_size, dust, line_opacity):
                self._native_emitters(slab_size, dust, line_opacity)
            for slab_size in {slab_size for _, slab_size in self._slabs}
            for dust, line_opacity in kinds}
        self._gmodel = driver.native_class('GModel', dtype)()

    def has_dust(self) -> bool:
        """Return whether it has components with dust."""
        return bool(self._plans.absorbers)

    def has_line_opacity(self) -> bool:
        """Return whether it has emitters with line opacity."""
        return any(layout.line_opacity for layout in self._layouts.values())

    def evaluate(
            self,
            params: ParamValues,
            outputs: dict[str, DeviceArray | None],
            out_extra: dict[str, Any] | None,
            opacity_front: DeviceArray | None,
            tau: DeviceArray | None
    ) -> LineResolution | None:
        """
        Add the light of the emitters that reaches the viewer to the
        output array, the 'image' or the 'scube' in outputs, dimmed by the
        opacity in front of the galaxy and its own, and add its own to that
        in front: the optical depth of the dust in front at each spaxel,
        in each cube of opacity (opacity_front, of shape (ncubes, ny, nx);
        required with dust of its own, None without dust in front or in
        it), and that of the lines at each sub-channel of each spaxel (tau,
        of shape (nchannels * subchannels, ny, nx); required with line
        opacity, None without line opacity in front or in it).

        The extra outputs of each component are named after its name, or
        its position (e.g. 'disk_bdata' or 'component0_bdata'), and the
        totals 'total_*': those of ComponentPlans.add_extras (along the
        depth axis), and of the shape of the output, 'total_unabsorbed',
        the light of the emitters without opacity (neither its own nor
        that in front), and with line opacity, 'total_line_tau', the
        optical depth of its lines at each channel (the mean of its
        sub-channels).

        With extras and line opacity, it returns how finely the march
        resolves it (see LineResolution); None otherwise.
        """
        plans = self._plans
        driver = plans.driver
        grid = plans.native_grid
        values = plans.params(params)
        extras = {} if out_extra is not None else None
        # (those of the volumes and the absorbers of each slab)
        slab_extras = []

        # The lines of the emitters, and the fields of the layers
        lines_h, lines_d = self._lines
        for i in plans.emitters:
            lines_h[self._rows[i]] = plans.plans[i].line_values(values[i])
            if self._layouts[i].layer:
                self._evaluate_fields(values, i, grid, extras)
        driver.mem_copy_h2d(lines_h, lines_d)

        image = outputs.get('image') is not None
        out = outputs['image'] if image else outputs['scube']
        dust = opacity_front is not None
        subchannels = self._subchannels if tau is not None else 1
        # With extras, the light without opacity, and the optical depth of
        # the lines in front of the galaxy
        unabsorbed = None
        tau_front = None
        resolution = None
        if out_extra is not None:
            unabsorbed = driver.mem_alloc_d(out.shape, plans.dtype)
            driver.mem_fill(unabsorbed, 0)
            if self.has_line_opacity():
                tau_front = driver.mem_copy_d2h(tau)
                resolution = _ResolutionTracker(
                    grid['spec_step'] / subchannels,
                    grid['spat_size'][1::-1])
        for z0, size in self._slabs:
            slab_grid = _slab_grid(grid, z0, size)
            extra = {} if extras is not None else None
            odata = None
            if self._odata is not None:
                odata = _view(
                    self._odata,
                    (plans.ncubes,) + slab_grid['spat_size'][::-1])
                plans.evaluate_opacity(values, slab_grid, odata, extra)
            for i in plans.emitters:
                if not self._layouts[i].layer:
                    self._evaluate_fields(values, i, slab_grid, extra)
                if resolution is not None and self._layouts[i].line_opacity:
                    resolution.add(i, driver.mem_copy_d2h(_view(
                        self._fields[i], self._layouts[i].shape(slab_grid))),
                        lines_h[self._rows[i], 1])
            self._march(
                slab_grid, image, self._emitters[size, dust, True],
                subchannels, odata, opacity_front, tau, out)
            if unabsorbed is not None:
                self._march(
                    slab_grid, image, self._emitters[size, False, False], 1,
                    None, None, None, unabsorbed)
            if extra is not None:
                slab_extras.append(extra)

        if out_extra is not None:
            # (the extras of the slabs, along the depth axis)
            for i in slab_extras[0]:
                extras.setdefault(i, {}).update({
                    key: np.concatenate(
                        [extra[i][key] for extra in slab_extras])
                    for key in slab_extras[0][i]})
            bdata = [
                extras[i]['bdata'] for i in plans.emitters
                if 'bdata' in extras.get(i, {})]
            plans.add_extras(
                out_extra, extras, dict(bdata=sum(bdata)) if bdata else {})
            out_extra['total_unabsorbed'] = plans.output_extra(
                driver.mem_copy_d2h(unabsorbed))
            if tau_front is not None:
                line_tau = driver.mem_copy_d2h(tau) - tau_front
                out_extra['total_line_tau'] = plans.output_extra(
                    line_tau.reshape(
                        (-1, subchannels) + line_tau.shape[1:]).mean(axis=1))
                return resolution.resolution(
                    line_tau.max(axis=0) >= _THIN_TAU)
        return None

    def _march(
            self,
            slab_grid: NativeGrid,
            image: bool,
            emitters: list[Any],
            subchannels: int,
            opacity: DeviceArray | None,
            opacity_front: DeviceArray | None,
            tau: DeviceArray | None,
            out: DeviceArray
    ) -> None:
        """
        March through the planes of a slab, adding the light of the given
        native emitters to the image or the spectral cube out (see the
        native GModel.transfer_march).
        """
        grid = self._plans.native_grid
        self._gmodel.transfer_march(
            emitters=emitters, spat_size=slab_grid['spat_size'],
            slab_zero=slab_grid['slab_zero'],
            depth_size=grid['spat_size'][2], image=image,
            spec_size=grid['spec_size'], spec_step=grid['spec_step'],
            spec_zero=grid['spec_zero'], subchannels=subchannels,
            opacity=opacity, opacity_front=opacity_front, tau=tau, out=out)

    def _evaluate_fields(
            self,
            values: dict[int, ParamValues],
            index: int,
            grid: NativeGrid,
            extras: dict[int, dict[str, Any]] | None
    ) -> None:
        """
        Write the fields of the emitter of the given index on the grid (the
        whole grid of a layer, or a slab of a volume); with extras, its
        extra outputs go to extras[index].
        """
        extra = {} if extras is not None else None
        self._plans.plans[index].evaluate_fields(
            values[index], grid,
            _view(self._fields[index], self._layouts[index].shape(grid)),
            extra)
        if extras is not None:
            extras.setdefault(index, {}).update(extra)

    def _native_emitters(
            self, size: int, dust: bool, line_opacity: bool
    ) -> list[Any]:
        """
        Return the native emitters of the march of a slab of the given
        number of planes: their fields, lines and, with dust, ratios; with
        line_opacity, those with line opacity have it.
        """
        plans = self._plans
        slab_grid = _slab_grid(plans.native_grid, 0, size)
        emitter_class = plans.driver.native_class(
            'TransferEmitter', plans.dtype)
        emitters = []
        for i in plans.emitters:
            layout = self._layouts[i]
            emitters.append(emitter_class(
                layer=layout.layer,
                line_opacity=line_opacity and layout.line_opacity,
                fields=_view(self._fields[i], layout.shape(
                    plans.native_grid if layout.layer else slab_grid)),
                lines=self._lines[1][self._rows[i]],
                opacity_ratios=self._ratios[i] if dust else None))
        return emitters


class _ResolutionTracker:
    """
    The resolution of the line opacity of a march at each spaxel (see
    LineResolution), of the voxels of the volumes with line opacity, slab
    by slab, for a width of the channels of the march and the shape (ny,
    nx) of the spaxels.
    """

    def __init__(self, channel: float, shape: tuple[int, int]):
        self._channel = channel
        self._spectral = np.zeros(shape)
        self._depth = np.zeros(shape)
        # (the last plane of the previous slab of each volume)
        self._last = {}

    def add(self, index: int, fields: np.ndarray, scales: np.ndarray) -> None:
        """
        Add the voxels of a slab of the volume of the given index: its
        fields (see FieldsLayout; with line opacity) and the scale of each
        of its lines (see ComponentPlan.line_values).
        """
        j, v, s = fields[:3]
        on = (j != 0) & (s > 0) & (fields[3:3 + len(scales)] != 0).any(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            # The narrowest line of each voxel, in the spectral units
            sigma = np.abs(scales).min() * s
            spectral = np.where(on, self._channel / sigma, 0)
            if index in self._last:
                v, s, on = (
                    np.concatenate([last[None], now])
                    for last, now in zip(self._last[index], (v, s, on)))
            pair = on[1:] & on[:-1]
            depth = np.where(
                pair,
                np.abs(v[1:] - v[:-1]) / np.minimum(s[1:], s[:-1]), 0)
        self._spectral = np.maximum(self._spectral, spectral.max(axis=0))
        self._depth = np.maximum(
            self._depth, depth.max(axis=0, initial=0))
        self._last[index] = (v[-1], s[-1], on[-1])

    def resolution(self, where: np.ndarray) -> LineResolution:
        """
        Return the worst resolution of the march at the given spaxels (a
        mask of shape (ny, nx)).
        """
        def worst(ratio):
            return float(np.max(ratio, where=where, initial=0))
        return LineResolution(worst(self._spectral), worst(self._depth))


def _slab_grid(grid: NativeGrid, z0: int, size: int) -> NativeGrid:
    """
    Return the grid of a slab of the given number of planes of a grid,
    from its plane z0 (see NativeGrid: every slab places a point, e.g. a
    cloud of a Monte Carlo disk, in the same plane).
    """
    return grid | dict(
        spat_size=grid['spat_size'][:2] + (size,), slab_zero=z0)


def _view(data: DeviceArray, shape: tuple[int, ...]) -> DeviceArray:
    """Return the first values of a flat array, as an array of a shape."""
    return data[:math.prod(shape)].reshape(shape)
