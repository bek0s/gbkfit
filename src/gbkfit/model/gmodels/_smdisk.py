
from . import _disk


__all__ = ['SMDisk', 'SMDiskPlan']


class SMDisk(_disk.Disk):

    def plan(self, driver, nlines, dtype):
        return SMDiskPlan(self, driver, nlines, dtype)


class SMDiskPlan(_disk.DiskPlan):

    def _impl_evaluate(self, params, grid_and_outputs, out_extra):
        self._backend.smdisk_evaluate(self._native_disk, **grid_and_outputs)
