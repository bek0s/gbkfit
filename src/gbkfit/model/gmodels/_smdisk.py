
from . import _disk


__all__ = ['SMDisk']


class SMDisk(_disk.Disk):

    def _impl_prepare(self, driver, dtype):
        pass

    def _impl_evaluate(self, driver, params, grid_and_outputs, out_extra):
        self._backend.smdisk_evaluate(self._native_disk, **grid_and_outputs)
