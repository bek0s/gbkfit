
import importlib
import logging


log = logging.getLogger(__name__)


def _register_factories(parser, factories):
    for factory in factories:
        try:
            mod_name = factory.rsplit('.', 1)[0]
            cls_name = factory.rsplit('.', 1)[1]
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            parser.register(cls)
        except Exception as e:
            log.warning(
                f"could not register factory for type {factory}; "
                f"{e.__class__.__name__}: {e}")


def _register_drivers():
    from gbkfit.driver.core import driver_parser as parser
    factories = [
        'gbkfit.driver.drivers.cuda.DriverCUDA',
        'gbkfit.driver.drivers.host.DriverHost']
    _register_factories(parser, factories)


_register_drivers()
