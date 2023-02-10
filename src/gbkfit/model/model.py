
import copy

from gbkfit.driver import driver_parser
from gbkfit.utils import iterutils, miscutils, parseutils, timeutils
from .core import dmodel_parser, gmodel_parser

__all__ = [
    'Model'
]


class Model(parseutils.BasicParserSupport):

    @classmethod
    def load(cls, info, *args, **kwargs):
        desc = parseutils.make_basic_desc(Model, 'model')
        parseutils.load_if_exists(driver_parser, info, 'drivers')
        parseutils.load_if_exists(dmodel_parser, info, 'dmodels')
        parseutils.load_if_exists(gmodel_parser, info, 'gmodels')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        return dict(
            drivers=driver_parser.dump(self.drivers()),
            dmodels=dmodel_parser.dump(self.dmodels())

        )



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
        self._items = tuple(item for item in zip(drivers, dmodels, gmodels))
        # Preallocate some data structures for convenience
        self._h_model_data = []
        for dmodel in dmodels:
            for key in dmodel.keys():
                self._h_model_data.append({key: dict(d=None, m=None, w=None)})
        self._d_model_data = copy.deepcopy(self._h_model_data)
        # Merge all pdescs into a dict and ensure they have unique keys
        self._pdescs, self._mappings = miscutils.merge_dicts_and_make_mappings(
            [gmodel.params() for gmodel in self.gmodels()], 'model')

    def nitems(self):
        return len(self.items())

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

    def model_d(self, params, out_extra=None):
        t = timeutils.SimpleTimer('model_eval').start()
        for i in range(self.nitems()):
            driver = self.drivers()[i]
            dmodel = self.dmodels()[i]
            gmodel = self.gmodels()[i]
            mapping = self._mappings[i]
            out_extra_i = {} if out_extra is not None else None
            self._d_model_data[i].update(
                dmodel.evaluate(
                    driver, gmodel,
                    {param: params[mapping[param]] for param in mapping},
                    out_extra_i))
            if out_extra is not None:
                for key, val in out_extra_i.items():
                    out_extra[f'model{i}_{key}'] = val
        t.stop()
        return self._d_model_data

    def model_h(self, params, out_extra=None):
        self.model_d(params, out_extra)
        t = timeutils.SimpleTimer('model_d2h').start()
        d_data = self._d_model_data
        h_data = self._h_model_data
        for i in range(self.nitems()):
            for key in h_data[i].keys():
                for k in h_data[i][key].keys():
                    if d_data[i][key][k] is not None:
                        h_data[i][key][k] = self.drivers()[i].mem_copy_d2h(
                            d_data[i][key][k], h_data[i][key][k])
        t.stop()
        return self._h_model_data


model_parser = parseutils.BasicParser(Model)
