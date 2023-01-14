
try:
    import gbkfit.driver.native.libgbkfit_cuda as native_module
except ModuleNotFoundError as e:
    raise RuntimeError(
        "the native cuda backend is not enabled in your gbkfit installation; "
        "the cuda driver will be disabled") from e

from .driver import DriverCuda
