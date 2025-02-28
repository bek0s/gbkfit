
from gbkfit.utils import parseutils


def _register_drivers():
    from gbkfit.driver.core import driver_parser as abstract_parser
    parsers = [
        'gbkfit.driver.drivers.cuda.DriverCuda',
        'gbkfit.driver.drivers.host.DriverHost',
        'gbkfit.driver.drivers.sycl.DriverSycl'
    ]
    parseutils.register_optional_parsers(abstract_parser, parsers, 'driver')


_register_drivers()
