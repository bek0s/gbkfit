
try:
    import gbkfit.native.libgbkfit_host
except ModuleNotFoundError:
    raise RuntimeError(
        "the host driver is not enabled in your gbkfit installation")

from .driver import DriverHost
