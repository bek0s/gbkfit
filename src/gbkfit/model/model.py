
from gbkfit.utils import iterutils, miscutils


__all__ = ['Model', 'ModelGroup', 'make_model_group']


class Model:

    def __init__(self, dmodel, gmodel, driver):
        self._driver = driver
        self._dmodel = dmodel
        self._gmodel = gmodel
        self._h_data = {key: dict(data=None, mask=None) for key in dmodel.onames()}

    def dmodel(self):
        return self._dmodel

    def gmodel(self):
        return self._gmodel

    def driver(self):
        return self._driver

    def pdescs(self):
        return self.gmodel().params()

    def evaluate_d(self, params, out_extra=None):
        return self._dmodel.evaluate(
            self._driver, self._gmodel, params, out_extra)

    def evaluate_h(self, params, out_extra=None):
        d_data = self.evaluate_d(params, out_extra)
        h_data = self._h_data
        for key in d_data:
            h_data[key]['data'] = self._driver.mem_copy_d2h(d_data[key]['data'], h_data[key]['data'])
            h_data[key]['mask'] = self._driver.mem_copy_d2h(d_data[key]['mask'], h_data[key]['mask'])
        return h_data


class ModelGroup:

    def __init__(self, models):
        self._models = models = iterutils.tuplify(models)
        self._pdescs, self._mappings = miscutils.merge_dicts_and_make_mappings(
            [model.pdescs() for model in models], 'model')

    def nmodels(self):
        return len(self._models)

    def models(self):
        return self._models

    def pdescs(self):
        return self._pdescs

    def evaluate_d(self, params, out_extra=None):
        d_data = []
        for model, mapping in zip(self._models, self._mappings):
            if out_extra is not None:
                out_extra.append({})
            d_data.append(model.evaluate_d(
                {param: params[mapping[param]] for param in mapping},
                out_extra[-1] if out_extra is not None else None))
        return d_data

    def evaluate_h(self, params, out_extra=None):
        h_data = []
        for model, mapping in zip(self._models, self._mappings):
            if out_extra is not None:
                out_extra.append({})
            h_data.append(model.evaluate_h(
                {param: params[mapping[param]] for param in mapping},
                out_extra[-1] if out_extra is not None else None))
        return h_data


def make_model_group(dmodels, gmodels, drivers):
    dmodels = iterutils.tuplify(dmodels)
    gmodels = iterutils.tuplify(gmodels)
    drivers = iterutils.tuplify(drivers)
    ndmodels = len(dmodels)
    ngmodels = len(gmodels)
    ndrivers = len(drivers)
    if not (ndmodels == ngmodels == ndrivers):
        raise RuntimeError(
            f"could not create model group; the number of "
            f"gmodels ({ngmodels}), "
            f"dmodels ({ndmodels}), and "
            f"drivers ({ndrivers}) must be equal")
    models = []
    for dmodel, gmodel, driver in zip(dmodels, gmodels, drivers):
        models.append(Model(dmodel, gmodel, driver))
    return ModelGroup(models)
