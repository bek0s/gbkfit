
try:
    import gbkfit.native.libgbkfit_cuda
except ModuleNotFoundError:
    raise RuntimeError(
        "the cuda driver is not enabled in your gbkfit installation")

from .driver import DriverCUDA
