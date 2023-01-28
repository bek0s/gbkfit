
from gbkfit.utils import parseutils


def _register_drivers():
    from gbkfit.driver.core import driver_parser as parser
    factories = [
        'gbkfit.driver.drivers.cuda.DriverCuda',
        'gbkfit.driver.drivers.host.DriverHost'
    ]
    parseutils.register_optional_parser_factories(parser, factories, 'driver')


_register_drivers()
