from gbkfit.driver.base import driver_parser
from .cuda import DriverCuda
from .host import DriverHost


# The drivers (each needs its native module, and cuda CuPy: see their
# requires)
driver_parser.register([
    DriverCuda,
    DriverHost,
])
