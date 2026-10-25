"""
The smooth disks: evaluated at every pixel or voxel of their grid.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from gbkfit.driver import Driver
from gbkfit.params import ParamValues
from ..base import FieldsLayout
from . import _disk


__all__ = [
    'SmoothDisk',
    'SmoothDiskPlan'
]


class SmoothDisk(_disk.NodalDisk):
    """A disk evaluated smoothly: on the pixels or voxels of its grid."""

    def plan(
            self,
            driver: Driver,
            nlines: int,
            dtype: np.dtype,
            opacity_rows: Sequence[int] | None = None
    ) -> _disk.DiskPlan:
        return SmoothDiskPlan(self, driver, nlines, dtype, opacity_rows)


class SmoothDiskPlan(_disk.DiskPlan):
    """The evaluation of a SmoothDisk."""

    def fields_layout(self) -> FieldsLayout:
        # (a thin disk is a layer where its plane crosses each line of
        # sight; a thick one fills the voxels with j, v, s and its line
        # opacity)
        if self._disk.traits('rht'):
            nk = self._fields_lines()
            return FieldsLayout(
                layer=False, nslots=3 + nk, line_opacity=nk > 0)
        return FieldsLayout(layer=True, nslots=4)

    def _impl_evaluate(
            self,
            params: ParamValues,
            grid_and_outputs: dict[str, Any],
            out_extra: dict[str, Any] | None
    ) -> None:
        self._backend.smdisk_evaluate(self._native_disk, **grid_and_outputs)
