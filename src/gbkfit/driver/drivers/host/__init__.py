
try:
    import gbkfit.driver.native._host as native_module
except ModuleNotFoundError as e:
    raise RuntimeError(
        "the native host backend is not enabled in your gbkfit installation; "
        "the host driver will be disabled") from e

from .driver import DriverHost
