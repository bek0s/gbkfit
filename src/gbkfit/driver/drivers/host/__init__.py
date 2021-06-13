
try:
    import gbkfit.driver.native.libgbkfit_host as native_module
except ModuleNotFoundError:
    raise RuntimeError(
        "the host driver is not enabled in your gbkfit installation")

from .driver import DriverHost
