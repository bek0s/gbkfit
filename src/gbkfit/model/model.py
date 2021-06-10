
from gbkfit.utils import iterutils, miscutils


__all__ = ['Model', 'ModelGroup', 'make_model_group', 'make_model_group_from_cmp', 'make_model_group_from_seq']


class Model_:

    def __init__(self, dmodels, gmodels, drivers):
        dmodels = iterutils.tuplify(dmodels)
        gmodels = iterutils.tuplify(gmodels)
        drivers = iterutils.tuplify(drivers)
        ndmodels = len(dmodels)
        ngmodels = len(gmodels)
        ndrivers = len(drivers)
        if not (ndmodels == ngmodels == ndrivers):
            raise RuntimeError(
                f"could not create model; the number of "
                f"gmodels ({ngmodels}), "
                f"dmodels ({ndmodels}), and "
                f"drivers ({ndrivers}) must be equal")
        models = []
        for dmodel, gmodel, driver in zip(dmodels, gmodels, drivers):
            models.append(Model(dmodel, gmodel, driver))


class Model:

    def __init__(self, dmodel, gmodel, driver):
        self._driver = driver
        self._dmodel = dmodel
        self._gmodel = gmodel
        self._h_data = {
            key: dict(d=None, m=None, w=None) for key in dmodel.onames()}

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
            for k in h_data[key].keys():
                if d_data[key][k] is not None:
                    h_data[key][k] = self._driver.mem_copy_d2h(
                        d_data[key][k], h_data[key][k])
        return h_data


class ModelGroup:

    def __init__(self, models):
        self._models = models = iterutils.tuplify(models)
        self._pdescs, self._mappings = miscutils.merge_dicts_and_make_mappings(
            [model.pdescs() for model in models], 'model')

    def __getitem__(self, index):
        return self._models[index]

    def __iter__(self):
        return iter(self._models)

    def __len__(self):
        return len(self._models)

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


def make_model_group_from_seq(models):
    return ModelGroup(iterutils.tuplify(models))


def make_model_group_from_cmp(dmodels, gmodels, drivers):
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
    return make_model_group_from_seq(models)
