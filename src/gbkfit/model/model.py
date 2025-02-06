
import abc
import copy
from typing import Any

from gbkfit.driver import Driver, driver_parser
from gbkfit.utils import iterutils, miscutils, parseutils, timeutils
from .core import DModel, GModel, dmodel_parser, gmodel_parser


__all__ = [
    'Model',
    'ModelGroup',
    'model_parser'
]


class Model(parseutils.BasicParserSupport, abc.ABC):

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'Model':
        dataset = kwargs.get('dataset')
        desc = parseutils.make_basic_desc(cls, 'model')
        parseutils.load_option_and_update_info(
            driver_parser, info, 'driver')
        parseutils.load_option_and_update_info(
            dmodel_parser, info, 'dmodel', dataset=dataset)
        parseutils.load_option_and_update_info(
            gmodel_parser, info, 'gmodel')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self) -> dict[str, Any]:
        return dict(
            driver=driver_parser.dump(self.driver()),
            gmodel=dmodel_parser.dump(self.dmodel()),
            dmodel=gmodel_parser.dump(self.gmodel()))

    def __init__(self, driver: Driver, dmodel: DModel, gmodel: GModel):
        self._driver = driver
        self._dmodel = dmodel
        self._gmodel = gmodel

    def pdescs(self):
        return self.gmodel().pdescs()

    def driver(self) -> Driver:
        return self._driver

    def dmodel(self) -> DModel:
        return self._dmodel

    def gmodel(self) -> GModel:
        return self._gmodel


model_parser = parseutils.BasicParser(Model)


class ModelGroup:

    def __init__(self, models: list[Model]):
        self._models = iterutils.tuplify(models)
        # Preallocate some data structures for convenience
        self._h_model_data = []
        for model in self.models():
            for key in model.dmodel().keys():
                self._h_model_data.append({key: dict(d=None, m=None, w=None)})
        self._d_model_data = copy.deepcopy(self._h_model_data)
        # Merge all pdescs into a dict and ensure they have unique keys
        self._pdescs, self._mappings = miscutils.merge_dicts_and_make_mappings(
            [model.gmodel().params() for model in self.models()], 'model')

    def pdescs(self):
        return self._pdescs

    def models(self) -> tuple[Model, ...]:
        return self._models

    def model_d(
            self,
            params: dict[str, Any],
            out_extra: dict[str, Any] = None
    ) -> list[dict[str, Any]]:
        t = timeutils.SimpleTimer('model_eval').start()
        for i, model in enumerate(self.models()):
            out_extra_i = {} if out_extra is not None else None
            driver = model.driver()
            dmodel = model.dmodel()
            gmodel = model.gmodel()
            mapping = self._mappings[i]
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

    def model_h(
            self,
            params: dict[str, Any],
            out_extra: dict[str, Any] = None
    ) -> list[dict[str, Any]]:
        self.model_d(params, out_extra)
        t = timeutils.SimpleTimer('model_d2h').start()
        d_data = self._d_model_data
        h_data = self._h_model_data
        for i, model in enumerate(self.models()):
            for key in h_data[i].keys():
                for k in h_data[i][key].keys():
                    if d_data[i][key][k] is not None:
                        h_data[i][key][k] = model.driver().mem_copy_d2h(
                            d_data[i][key][k], h_data[i][key][k])
        t.stop()
        return self._h_model_data
