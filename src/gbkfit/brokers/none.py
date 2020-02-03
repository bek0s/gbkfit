
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
        self._params = None
        self._dextra = None
        self._gextra = None

    def _prepare_impl(self):
        pass

    def _evaluate_impl(self, params, dextra, gextra):
        self._params = params
        self._dextra = dextra
        self._gextra = gextra

    def output(self):
        output_dextra = {}
        output_gextra = {}
        output = self._dmodel.evaluate(
            self._driver, self._gmodel, self._params,
            output_dextra if self._dextra else None,
            output_gextra if self._gextra else None)
        return output, output_dextra, output_gextra
