"""
The evaluation of a galaxy on the grid of an observation (GalaxyPlan).
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from gbkfit.driver import DeviceArray
from gbkfit.params import ParamValues
from ._component_plans import ComponentPlans
from ._transfer import LineResolution
from .components.base import (
    ComponentState, DepthAxis, OpacityCubes, SceneTarget)
from .extras import Extras

if TYPE_CHECKING:
    from .galaxy import Galaxy


__all__ = [
    'GalaxyPlan'
]


class GalaxyPlan:
    """
    The evaluation of a galaxy on the grid of an observation, the additive
    counterpart of a march of radiative transfer (see TransferPlan): of
    its components (see ComponentPlans), in two stages: the optical depth of
    the absorbers in each voxel, then the light of the emitters, dimmed by
    the optical depth from the viewer to each voxel.

    The parameters are those of ComponentPlans; the target gives the
    spectral smoothing of the instrument (km/s; see
    ComponentPlans.light_resolution; None for an image).
    """

    def __init__(
            self,
            galaxy: 'Galaxy',
            target: SceneTarget,
            emitters: Sequence[int],
            cubes: OpacityCubes,
            depth: DepthAxis | None
    ):
        plans = ComponentPlans(galaxy, target, depth, emitters, cubes)
        self._plans = plans
        self._depth = depth
        self._smoothing = target.smoothing
        # The cubes of opacity
        self._odata = None
        if plans.ncubes:
            self._odata = target.driver.mem_alloc_d(
                (plans.ncubes,) + plans.native_grid['spat_size'][::-1],
                target.dtype)
            self._gmodel = target.driver.native_class(
                'GModel', target.dtype)()

    def evaluate(
            self,
            params: ParamValues,
            states: dict[int, ComponentState],
            outputs: dict[str, DeviceArray | None],
            out_extra: Extras | None
    ) -> LineResolution | None:
        """
        Add the light of the emitters to the output array, the 'image' or
        the 'scube' in outputs, with the parameters of the galaxy and what
        it gives its components, their states (see GalaxyStage.evaluate).
        The extra
        outputs of each component have its name, or its position, as the
        id 'component' (e.g. 'disk' or 'component0'), and their totals
        none (see ComponentPlans.add_extras). With extra outputs, a
        warning says where zmax shapes a disk (see
        ComponentPlans.warn_truncation), and it returns how finely it
        resolves the lines (see LineResolution: their light only); None
        otherwise.
        """
        plans = self._plans
        driver = plans.driver
        dtype = plans.dtype
        grid = plans.native_grid
        shape = grid['spat_size'][::-1]
        odata = self._odata
        extras = {} if out_extra is not None else None
        bdata = None
        obdata = None
        if out_extra is not None:
            bdata = driver.mem_alloc_d(shape, dtype)
            driver.mem_fill(bdata, 0)
            if self._depth is not None:
                obdata = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(obdata, 0)
        values = plans.params(params)
        if out_extra is not None:
            plans.warn_truncation(values)

        # The absorbers write the optical depth of each voxel, which then
        # becomes the optical depth from the viewer to the far side of each
        # voxel, which the emitters read
        if odata is not None:
            driver.mem_fill(odata, 0)
            plans.evaluate_opacity(values, states, grid, odata, extras)
            for cube in odata:
                self._gmodel.opacity_accumulate(
                    spat_size=grid['spat_size'], opacity=cube)

        outputs = outputs | dict(bdata=bdata, odata=odata, obdata=obdata)
        for i in plans.emitters:
            extra = {} if extras is not None else None
            plans.plans[i].evaluate_emission(
                values[i], grid, outputs, extra, states.get(i))
            if extras is not None:
                extras.setdefault(i, {}).update(extra)

        if out_extra is None:
            return None
        plans.add_extras(out_extra, extras, {
            name: driver.mem_copy_d2h(total)
            for name, total in dict(bdata=bdata, obdata=obdata).items()
            if total is not None})
        light = 0.0 if self._smoothing is None \
            else plans.light_resolution(values, extras, self._smoothing)
        return LineResolution(0.0, 0.0, light)
