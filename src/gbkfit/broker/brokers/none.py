
import gbkfit.broker


class BrokerNone(gbkfit.broker.Broker):

    @staticmethod
    def type():
        return 'none'

    @classmethod
    def load(cls, info):
        return cls()

    def dump(self):
        return None

    def __init__(self):
        super().__init__()

    def _impl_prepare(self, driver, dmodel, gmodel):
        pass

    def _impl_evaluate(
            self, driver, dmodel, gmodel, params, out_dextra, out_gextra):
        return dmodel.evaluate(driver, gmodel, params, out_dextra, out_gextra)
