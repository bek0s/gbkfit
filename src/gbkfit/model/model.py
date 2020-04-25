"""
import gbkfit.model.brokers
import gbkfit.model.drivers


class Model:

    def __init__(self, dmodel, gmodel, driver, broker):
        if not driver:
            driver = gbkfit.model.drivers.DriverHost()
        if not broker:
            broker = gbkfit.model.brokers.BrokerNone()
        self._dmodel = dmodel
        self._gmodel = gmodel
        self._driver = driver
        self._broker = broker

    def dmodel(self):
        return self._dmodel

    def gmodel(self):
        return self._gmodel

    def driver(self):
        return self._driver

    def broker(self):
        return self._broker

    def params(self):
        return self._gmodel.params()

    def evaluate(self, params, out_dextra=None, out_gextra=None):
        dmodel = self._dmodel
        gmodel = self._gmodel
        driver = self._driver
        broker = self._broker
        return broker.evaluate(
            driver, dmodel, gmodel, params, out_dextra, out_gextra)



class ModelEvaluator:

    def __init__(self, dmodels):
        pass
"""
