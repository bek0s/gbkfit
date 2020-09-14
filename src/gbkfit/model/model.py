
from gbkfit.utils import iterutils, miscutils


__all__ = ['Model']


class Model:

    def __init__(self, dmodels, gmodels, drivers):
        dmodels = iterutils.tuplify(dmodels)
        gmodels = iterutils.tuplify(gmodels)
        drivers = iterutils.tuplify(drivers)
        ndmodels = len(dmodels)
        ngmodels = len(gmodels)
        ndrivers = len(drivers)
        self._drivers = drivers
        self._dmodels = dmodels
        self._gmodels = gmodels
        if not (ndmodels == ngmodels == ndrivers):
            raise RuntimeError(
                f"the number of "
                f"gmodels ({ngmodels}), "
                f"dmodels ({ndmodels}), and "
                f"drivers ({ndrivers}) must be equal")
        # ...
        pdescs, mappings = miscutils.merge_dicts_and_make_mappings(
            [g.params() for g in gmodels], 'model')
        self._pdescs = pdescs
        self._mappings = mappings
        # ...
        self._h_data = [{key: None for key in mdl.onames()} for mdl in dmodels]

    def nitems(self):
        return len(self.dmodels())

    def dmodels(self):
        return self._dmodels

    def gmodels(self):
        return self._gmodels

    def drivers(self):
        return self._drivers

    def pdescs(self):
        return self._pdescs

    def evaluate_d(self, params, out_extra=None):
        d_data = []
        for i in range(self.nitems()):
            dmodel = self.dmodels()[i]
            gmodel = self.gmodels()[i]
            driver = self.drivers()[i]
            mapping = self._mappings[i]
            if out_extra is not None:
                out_extra.append({})
            d_data.append(dmodel.evaluate(
                driver, gmodel,
                {param: params[mapping[param]] for param in mapping},
                out_extra[-1] if out_extra is not None else None))
        return d_data

    def evaluate_h(self, params, out_extra=None):
        d_data = self.evaluate_d(params, out_extra)
        h_data = self._h_data
        for i in range(self.nitems()):
            for key in d_data[i]:
                h_data[i][key] = self._drivers[i].mem_copy_d2h(
                    d_data[i][key], h_data[i][key])
        return h_data
