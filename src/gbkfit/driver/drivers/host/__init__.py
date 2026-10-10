try:
    import gbkfit.driver.native._host as native_module
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "the host driver needs the native host module, which this "
        "installation of gbkfit was built without") from e

from .driver import DriverHost
