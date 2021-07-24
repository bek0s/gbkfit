
import collections
import copy

import time

from gbkfit.utils import iterutils, miscutils
from . import _detail


class ObjectiveModel:

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
        self._h_model_data = [
            {key: dict(d=None, m=None, w=None) for key in dmodel.onames()}
            for dmodel in self.dmodels()]
        self._d_model_data = copy.deepcopy(self._h_model_data)
        # Merge the pdescs of all models into a dict and ensure they
        # have unique keys
        self._pdescs, self._mappings = miscutils.merge_dicts_and_make_mappings(
            [gmodel.params() for gmodel in self.gmodels()], 'model')
        self._times_mdl_eval = []
        self._times_mdl_d2h = []
        self._time_samples = collections.defaultdict(list)

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
        return _detail.time_stats(self.time_stats_samples(False))

    def time_stats_reset(self):
        for samples in self.time_stats_samples(False).values():
            samples.clear()

    def time_stats_samples(self, copy_=True):
        return copy.deepcopy(self._time_samples) \
            if copy_ else self._time_samples

    def model_d(self, params, out_extra=None):
        t1 = time.time_ns()
        for i in range(self.nitems()):
            driver = self.drivers()[i]
            dmodel = self.dmodels()[i]
            gmodel = self.gmodels()[i]
            mapping = self._mappings[i]
            out_extra_i = dict()
            self._d_model_data[i].update(dmodel.evaluate(
                driver, gmodel,
                {param: params[mapping[param]] for param in mapping},
                out_extra_i if out_extra is not None else None))
            for key, val in out_extra_i.items():
                prefix = f'{i}_' * bool(self.nitems() > 1)
                out_extra[f'{prefix}{key}'] = val
        t2 = time.time_ns()
        self.time_stats_samples(False)['mdl_eval'].append(t2 - t1)
        return self._d_model_data

    def model_h(self, params, out_extra=None):
        self.model_d(params, out_extra)
        t1 = time.time_ns()
        d_data = self._d_model_data
        h_data = self._h_model_data
        for i in range(self.nitems()):
            for key in h_data[i].keys():
                for k in h_data[i][key].keys():
                    if d_data[i][key][k] is not None:
                        h_data[i][key][k] = self.drivers()[i].mem_copy_d2h(
                            d_data[i][key][k], h_data[i][key][k])
        t2 = time.time_ns()
        self.time_stats_samples(False)['mdl_d2h'].append(t2 - t1)
        return self._h_model_data
