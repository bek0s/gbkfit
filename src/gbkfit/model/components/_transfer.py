"""
The evaluation of the components of a galaxy by radiative transfer along
each line of sight (TransferPlan; see the line of sight Transfer).
"""

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamValues
from gbkfit.utils import gridutils
from ._component_plans import ComponentPlans
from .base import DepthAxis, NativeGrid
from .lines import Band

if TYPE_CHECKING:
    from ._component_set import ComponentSet


__all__ = [
    'TransferPlan'
]


class TransferPlan:
    """
    The evaluation of the components of a galaxy on the grid of an
    observation by radiative transfer along each line of sight: the
    absorbers write the optical depth of each voxel to the cubes of
    opacity, the emitters write the fields of their light (see
    ComponentPlan.evaluate_fields), and a march along each line of sight,
    front to back, adds the light of each channel that reaches the viewer
    to the image or the spectral cube.

    It evaluates the depth axis in slabs of planes, one after the other,
    to bound its memory: the absorbers and the volumes (e.g. thick disks)
    on the planes of each slab (the clouds of a Monte Carlo disk are drawn
    again for each, the same ones), the layers (e.g. thin disks, points)
    once. The parameters are those of ComponentPlans, with a depth axis,
    and the number of planes of a slab (slab; all if None).
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
            lines: tuple[str, ...] | None,
            band: Band | None,
            redshift: float,
            slab: int | None
    ):
        plans = ComponentPlans(
            component_set, driver, grid, spectral, depth, dtype, emitters,
            lines, band, redshift)
        self._plans = plans
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
        # The cubes of opacity of a slab (flat, as the fields), the optical
        # depth of the dust in front of the slab, and with line opacity,
        # that of the lines at each channel
        self._odata = None
        self._opacity_front = None
        if plans.ncubes:
            self._odata = driver.mem_alloc_d(
                plans.ncubes * math.prod(slab_grid['spat_size']), dtype)
            self._opacity_front = driver.mem_alloc_d(
                (plans.ncubes,) + native_grid['spat_size'][1::-1], dtype)
        self._tau = None
        if any(layout.line_opacity for layout in self._layouts.values()):
            self._tau = driver.mem_alloc_d(
                (native_grid['spec_size'],)
                + native_grid['spat_size'][1::-1], dtype)
        # The emitters of the march of slabs of each size
        self._emitters = {
            slab_size: self._native_emitters(slab_size)
            for slab_size in {slab_size for _, slab_size in self._slabs}}
        self._gmodel = driver.native_class('GModel', dtype)()

    def evaluate(
            self,
            params: ParamValues,
            outputs: dict[str, DeviceArray | None],
            out_extra: dict[str, Any] | None
    ) -> None:
        """
        Add the light of the emitters that reaches the viewer to the
        output array, the 'image' or the 'scube' in outputs. The extra
        outputs of each component are named after its name, or its
        position (e.g. 'disk_bdata' or 'component0_bdata'), and the totals
        'total_*'.
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

        image = outputs.get('image')
        for state in (self._opacity_front, self._tau):
            if state is not None:
                driver.mem_fill(state, 0)
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
            self._gmodel.transfer_march(
                emitters=self._emitters[size],
                spat_size=slab_grid['spat_size'], slab_zero=z0,
                depth_size=grid['spat_size'][2],
                image=image is not None, spec_size=grid['spec_size'],
                spec_step=grid['spec_step'], spec_zero=grid['spec_zero'],
                opacity=odata, opacity_front=self._opacity_front,
                tau=self._tau,
                out=image if image is not None else outputs['scube'])
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

    def _native_emitters(self, size: int) -> list[Any]:
        """
        Return the native emitters of the march of a slab of the given
        number of planes: their fields, lines and ratios.
        """
        plans = self._plans
        slab_grid = _slab_grid(plans.native_grid, 0, size)
        emitter_class = plans.driver.native_class(
            'TransferEmitter', plans.dtype)
        emitters = []
        for i in plans.emitters:
            layout = self._layouts[i]
            emitters.append(emitter_class(
                layer=layout.layer, line_opacity=layout.line_opacity,
                fields=_view(self._fields[i], layout.shape(
                    plans.native_grid if layout.layer else slab_grid)),
                lines=self._lines[1][self._rows[i]],
                opacity_ratios=self._ratios[i]))
        return emitters


def _slab_grid(grid: NativeGrid, z0: int, size: int) -> NativeGrid:
    """
    Return the grid of a slab of the given number of planes of a grid,
    from its plane z0.
    """
    step = grid['spat_step'][2]
    return grid | dict(
        spat_size=grid['spat_size'][:2] + (size,),
        spat_zero=grid['spat_zero'][:2] + (grid['spat_zero'][2] + z0 * step,))


def _view(data: DeviceArray, shape: tuple[int, ...]) -> DeviceArray:
    """Return the first values of a flat array, as an array of a shape."""
    return data[:math.prod(shape)].reshape(shape)
