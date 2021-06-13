
try:
    import gbkfit.driver.native.libgbkfit_cuda as native_module
except ModuleNotFoundError as e:
    raise RuntimeError(
        "the cuda driver is not enabled in your gbkfit installation") from e

from .driver import DriverCUDA
