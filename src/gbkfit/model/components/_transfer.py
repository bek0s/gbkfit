"""
The evaluation of the components of a galaxy by radiative transfer along
each line of sight (TransferPlan; see the line of sight Transfer).
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamValues
from gbkfit.utils import gridutils
from ._component_plans import ComponentPlans
from .base import DepthAxis
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

    The parameters are those of ComponentPlans; it needs a depth axis.
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
            redshift: float
    ):
        plans = ComponentPlans(
            component_set, driver, grid, spectral, depth, dtype, emitters,
            lines, band, redshift)
        native_grid = plans.native_grid
        components = component_set.components()
        # The fields of each emitter, and its lines (in one buffer, which
        # is copied to the device once per evaluation): one row for each
        # of its wavelengths (those of its lines, or its band)
        nrows = [
            len(components[i].emission().wavelengths(lines, band, redshift))
            for i in plans.emitters]
        ends = np.cumsum(nrows)
        self._rows = {
            i: slice(end - n, end)
            for i, n, end in zip(plans.emitters, nrows, ends)}
        self._lines = driver.mem_alloc_s((sum(nrows), 3), dtype)
        self._fields = {}
        emitter_class = driver.native_class('TransferEmitter', dtype)
        self._emitters = []
        for i in plans.emitters:
            layout = plans.plans[i].fields_layout()
            self._fields[i] = driver.mem_alloc_d(
                layout.shape(native_grid), dtype)
            ratios = plans.ratios[i]
            self._emitters.append(emitter_class(
                layer=layout.layer, fields=self._fields[i],
                lines=self._lines[1][self._rows[i]],
                opacity_ratios=None if ratios is None
                else driver.mem_copy_h2d(ratios)))
        # The optical depth of the dust in front of the slab
        self._opacity_front = None
        if plans.odata is not None:
            self._opacity_front = driver.mem_alloc_d(
                plans.odata.shape[:1] + plans.odata.shape[2:], dtype)
        self._plans = plans
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
        totals = {}

        tau = plans.evaluate_opacity(values, out_extra)
        if tau is not None:
            totals['odata'] = tau

        lines_h, lines_d = self._lines
        for i in plans.emitters:
            extra = {} if out_extra is not None else None
            plans.plans[i].evaluate_fields(
                values[i], grid, self._fields[i], extra)
            lines_h[self._rows[i]] = plans.plans[i].line_values(values[i])
            if extra is not None and 'bdata' in extra:
                totals['bdata'] = totals['bdata'] + extra['bdata'] \
                    if 'bdata' in totals else extra['bdata']
            plans.add_extra(out_extra, i, extra)
        driver.mem_copy_h2d(lines_h, lines_d)

        image = outputs.get('image')
        if self._opacity_front is not None:
            driver.mem_fill(self._opacity_front, 0)
        self._gmodel.transfer_march(
            emitters=self._emitters, spat_size=grid['spat_size'],
            slab_zero=0, depth_size=grid['spat_size'][2],
            image=image is not None, spec_size=grid['spec_size'],
            spec_step=grid['spec_step'], spec_zero=grid['spec_zero'],
            opacity=plans.odata, opacity_front=self._opacity_front,
            out=image if image is not None else outputs['scube'])

        if out_extra is not None:
            for name, total in totals.items():
                out_extra[f'total_{name}'] = plans.extra(total)
