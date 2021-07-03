
import numbers
import time

import numpy as np

from gbkfit.utils import iterutils
from .residual import ObjectiveResidual
from . import _detail


class ObjectiveGoodness(ObjectiveResidual):

    def __init__(
            self, datasets, drivers, dmodels, gmodels,
            wd=False, wp=0.0, wu=1.0):
        super().__init__(datasets, drivers, dmodels, gmodels)
        n = self.nitems()
        self._s_residual_scalar = iterutils.make_list(n, (None, None))
        self._s_counts = iterutils.make_list(n, (None, None))
        self._backends = iterutils.make_list(n, None)
        if not iterutils.is_sequence(wp):
            wp = iterutils.make_tuple(n, wp)
        if not iterutils.is_sequence(wu):
            wu = iterutils.make_tuple(n, wu)
        self._wd = wd
        self._wp = wp
        self._wu = wu
        self._weights_d = iterutils.make_tuple(n, dict())
        self._weights_p = iterutils.make_tuple(n, dict())
        self._weights_u = iterutils.make_tuple(n, dict())
        if len(wp) != n:
            raise RuntimeError(
                f"the length of wp and the number of datasets are not equal "
                f"({len(wp)} != {n})")
        if len(wu) != n:
            raise RuntimeError(
                f"the length of wu and the number of datasets are not equal "
                f"({len(wu)} != {n})")
        for i in range(n):
            for name in self.dmodels()[i].onames():
                # ...
                min_ = np.nanmin(self.datasets()[i][name].data())
                max_ = np.nanmax(self.datasets()[i][name].data())
                self._weights_d[i][name] = 1 / (max_ - min_) if wd else 1.0
                # ...
                if isinstance(wp[i], type(None)):
                    self._weights_p[i][name] = 0.0
                elif isinstance(wp[i], numbers.Real):
                    self._weights_p[i][name] = wp[i]
                elif isinstance(wp[i], dict):
                    self._weights_p[i][name] = wp[i].get(name, 0.0)
                # ...
                if isinstance(wu[i], type(None)):
                    self._weights_u[i][name] = 1.0
                elif isinstance(wu[i], numbers.Real):
                    self._weights_u[i][name] = wu[i]
                elif isinstance(wu[i], dict):
                    self._weights_u[i][name] = wu[i].get(name, 1.0)
        self._times_gds_eval_vector = []
        self._times_gds_eval_scalar = []
        self._prepared = False

    def time_stats(self):
        time_stats = super().time_stats()
        return time_stats | _detail.time_stats(dict(
            gds_eval_vector=self._times_gds_eval_vector,
            gds_eval_scalar=self._times_gds_eval_scalar))

    def time_stats_reset(self):
        super().time_stats_reset()
        self._times_gds_eval_vector.clear()
        self._times_gds_eval_scalar.clear()

    def prepare(self):
        for i, (driver, dmodel) in zip(self.drivers(), self.dmodels()):
            self._s_residual_scalar[i] = driver.mem_alloc_s(1, dmodel.dtype())
            self._s_counts[i] = driver.mem_alloc_s(3, np.int32)
            self._backends[i] = driver.backend().make_objective(dmodel.dtype())
        self._prepared = True

    def goodness_residual_nddata(self, params, out_extra=None):
        self._evaluate_goodness_residual_d(params, out_extra)
        return self.residual_nddata_h_data()

    def goodness_residual_vector(self, params, out_extra=None):
        self._evaluate_goodness_residual_d(params, out_extra)
        return self.residual_vector_h_data()

    def goodness_residual_scalar(self, params, out_extra=None):
        self._evaluate_goodness_residual_d(params, out_extra)
        residuals = []
        for i, driver in enumerate(self.drivers()):
            d_residual_vector = self.residual_vector_d_data()[i]
            h_residual_scalar = self._s_residual_scalar[i][0]
            d_residual_scalar = self._s_residual_scalar[i][1]
            driver.math_abs(d_residual_vector, out=d_residual_vector)
            driver.math_sum(d_residual_vector, out=d_residual_scalar)
            driver.mem_copy_d2h(d_residual_scalar, h_residual_scalar)
            residuals.append(h_residual_scalar[0])
        return residuals

    def _evaluate_goodness_residual_d(self, params, out_extra=None):
        # Evaluate residual
        self.residual_vector_d(params, out_extra)
        # Apply weights
        pass    # TODO
