
import copy

import time

from gbkfit.utils import iterutils, miscutils
from . import _detail


class ObjectiveEvaluate:

    def __init__(self, drivers, dmodels, gmodels):
        drivers = iterutils.tuplify(drivers)
        dmodels = iterutils.tuplify(dmodels)
        gmodels = iterutils.tuplify(gmodels)
        ndrivers = len(drivers)
        ndmodels = len(dmodels)
        ngmodels = len(gmodels)
        if not (ndrivers == ndmodels == ngmodels):
            raise RuntimeError(
                f"the number of "
                f"drivers ({ndrivers}), "
                f"dmodels ({ndmodels}), and "
                f"gmodels ({ngmodels}) must be equal")
        self._drivers = drivers
        self._dmodels = dmodels
        self._gmodels = gmodels
        self._items = tuple([
            (drivers[i], dmodels[i], gmodels[i]) for i in range(self.nitems())])
        # Preallocate some data structures
        # It will make our life easier later on
        self._h_data = [
            {key: dict(d=None, m=None, w=None) for key in dmodel.onames()}
            for dmodel in self.dmodels()]
        self._d_data = copy.deepcopy(self._h_data)
        # Merge the pdescs of all models into a dict and ensure they
        # have unique keys
        self._pdescs, self._mappings = miscutils.merge_dicts_and_make_mappings(
            [model.pdescs() for model in self.gmodels()], 'model')
        self._times_mdl_eval = []
        self._times_mdl_d2h = []

    def nitems(self):
        return len(self.dmodels())

    def items(self):
        return self._items

    def drivers(self):
        return self._drivers

    def dmodels(self):
        return self._dmodels

    def gmodels(self):
        return self._gmodels

    def pdescs(self):
        return self._pdescs

    def time_stats(self):
        return _detail.time_stats(dict(
            mdl_eval=self._times_mdl_eval,
            mdl_d2h=self._times_mdl_d2h))

    def time_stats_reset(self):
        self._times_mdl_eval.clear()
        self._times_mdl_d2h.clear()

    def evaluate_d(self, params, out_extra=None):
        t1 = time.time_ns()
        for i in range(self.nitems()):
            driver = self.drivers()[i]
            dmodel = self.dmodels()[i]
            gmodel = self.gmodels()[i]
            mapping = self._mappings[i]
            if out_extra is not None:
                out_extra.append({})
            self._d_data[i].update(dmodel.evaluate(
                driver, gmodel,
                {param: params[mapping[param]] for param in mapping},
                out_extra[-1] if out_extra is not None else None))
        t2 = time.time_ns()
        self._times_mdl_eval.append(t2 - t1)
        return self._d_data

    def evaluate_h(self, params, out_extra=None):
        self.evaluate_d(params, out_extra)
        t1 = time.time_ns()
        d_data = self._d_data
        h_data = self._h_data
        for i in range(self.nitems()):
            for key in h_data[i].keys():
                for k in h_data[i][key].keys():
                    if d_data[i][key][k] is not None:
                        h_data[i][key][k] = self.drivers()[i].mem_copy_d2h(
                            d_data[i][key][k], h_data[i][key][k])
        t2 = time.time_ns()
        self._times_mdl_d2h.append(t2 - t1)
        return self._h_data
