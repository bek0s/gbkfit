
try:
    import gbkfit.driver.native.libgbkfit_sycl as native_module
except ModuleNotFoundError as e:
    raise RuntimeError(
        "the native sycl backend is not enabled in your gbkfit installation; "
        "the sycl driver will be disabled") from e

from .driver import DriverSycl
