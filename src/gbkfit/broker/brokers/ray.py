
try:
    import ray
except ImportError:
    raise ImportError(
        "To use the Ray-based broker, please install Ray.")

import numpy as np

import gbkfit.broker
from gbkfit.utils import parseutils
from . import _detail


@ray.remote
class _Actor:

    def __init__(self, driver, dmodel, gmodel):
        self._driver = driver
        self._dmodel = dmodel
        self._gmodel = gmodel

    def evaluate(self, params, request_dextra, request_gextra):
        out_dextra = {} if request_dextra else None
        out_gextra = {} if request_gextra else None
        output = self._dmodel.evaluate(
            self._driver, self._gmodel, params, out_dextra, out_gextra)
        return output, out_dextra, out_gextra


class BrokerRay(gbkfit.broker.Broker):

    @staticmethod
    def type():
        return 'ray'

    @classmethod
    def load(cls, info):
        return cls(**parseutils.parse_class_args(cls, info))

    def dump(self):
        return dict(
            grid=self._grid,
            address=self._address)

    def __init__(self, grid=(1, 1, 1), address=None):
        super().__init__()
        self._grid = tuple(grid)
        self._address = address
        self._ranges = []
        self._tiles = []
        self._futures = []
        self._output = {}
        if not ray.is_initialized():
            ray.init(address)

    def _impl_prepare(self, driver, dmodel, gmodel):
        size = dmodel.size()
        dtype = dmodel.dtype()
        onames = dmodel.onames()
        self._ranges = _detail.make_ranges(size, self._grid)
        for range_nd in self._ranges:
            idx_min = list(zip(*range_nd))[0]
            idx_max = list(zip(*range_nd))[1]
            dmodel_tile = dmodel.submodel(idx_min, idx_max)
            self._tiles.append(_Actor.remote(driver, dmodel_tile, gmodel))
        self._output = {oname: np.empty(size[::-1], dtype) for oname in onames}

    def _impl_evaluate(
            self, driver, dmodel, gmodel, params, out_dextra, out_gextra):
        self._futures = []
        for tile in self._tiles:
            self._futures.append(tile.evaluate.remote(
                params, out_dextra is not None, out_gextra is not None))
        for range_nd, future in zip(self._ranges, self._futures):
            suboutput, subout_dextra, subout_gextra = ray.get(future)
            for key, value in suboutput.items():
                slice_nd = _detail.make_range_slice(range_nd)[::-1]
                self._output[key][slice_nd] = value
            if subout_dextra:
                out_dextra.update(_detail.rename_extra(subout_dextra, range_nd))
            if subout_gextra:
                out_gextra.update(_detail.rename_extra(subout_gextra, range_nd))
        return self._output
