
import time

from gbkfit.utils import iterutils, parseutils
from .residual import ObjectiveResidual


class ObjectiveGoodness(ObjectiveResidual):

    @classmethod
    def load(cls, info, datasets, drivers, dmodels, gmodels):
        return super().load(info, datasets, drivers, dmodels, gmodels)

    def dump(self):
        return super().dump()

    def __init__(
            self, datasets, drivers, dmodels, gmodels,
            wd=False, wp=0.0, wu=1.0):
        super().__init__(datasets, drivers, dmodels, gmodels, wd, wp, wu)
        n = self.nitems()
        self._s_residual_scalar = iterutils.make_list(n, (None, None))
        self._backends = iterutils.make_list(n, None)
        self._prepared = False

    def prepare(self):
        super().prepare()
        for i in range(self.nitems()):
            driver = self.drivers()[i]
            dmodel = self.dmodels()[i]
            self._s_residual_scalar[i] = driver.mem_alloc_s(1, dmodel.dtype())
            self._backends[i] = driver.backend().make_objective(dmodel.dtype())
        self._prepared = True

    def residual_scalar(self, params, out_extra=None):
        t1 = time.time_ns()
        d_residual_vectors = self.residual_vector_d(params, out_extra)
        residuals = []
        for i, driver in enumerate(self.drivers()):
            d_residual_vector = d_residual_vectors[i]
            h_residual_scalar = self._s_residual_scalar[i][0]
            d_residual_scalar = self._s_residual_scalar[i][1]
            driver.math_abs(d_residual_vector, out=d_residual_vector)
            driver.math_sum(d_residual_vector, out=d_residual_scalar)
            driver.mem_copy_d2h(d_residual_scalar, h_residual_scalar)
            residuals.append(h_residual_scalar[0])
        t2 = time.time_ns()
        self.time_stats_samples(False)['gds_scalar'].append(t2 - t1)
        return residuals

    def log_likelihood(self, params, out_extra=None):
        t1 = time.time_ns()
        d_residual_vectors = self.residual_vector_d(params, out_extra)
        log_likelihoods = []
        for i, driver in enumerate(self.drivers()):
            # TODO
            d_residual_vector = d_residual_vectors[i]
            h_residual_scalar = self._s_residual_scalar[i][0]
            d_residual_scalar = self._s_residual_scalar[i][1]
            driver.math_mul(d_residual_vector, d_residual_vector, out=d_residual_vector)
            driver.math_sum(d_residual_vector, out=d_residual_scalar)
            driver.mem_copy_d2h(d_residual_scalar, h_residual_scalar)
            log_likelihoods.append(-0.5*h_residual_scalar[0])
        t2 = time.time_ns()
        self.time_stats_samples(False)['gds_loglike'].append(t2 - t1)
        return log_likelihoods


goodness_objective_parser = parseutils.BasicParser(ObjectiveGoodness)
