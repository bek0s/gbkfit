try:
    import gbkfit.driver.native._cuda as native_module
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "the cuda driver needs the native cuda module, which this "
        "installation of gbkfit was built without") from e
try:
    import cupy
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "the cuda driver needs CuPy: pip install 'gbkfit[cuda]'") from e

from .driver import DriverCuda
