
import numpy as np
import lmfit

import gbkfit.fitter


class FitterLMFit(gbkfit.fitter.Fitter):

    @staticmethod
    def type():
        return 'lmfit'

    @classmethod
    def load(cls, info):
        method = info.get("method", "leastsq")
        return cls(method)

    def dump(self):
        return {'type': self.type()}

    def __init__(
            self,
            method="leastsq",
            scale_covar=False,
            nan_policy="raise",
            reduce_fcn=None,
            **kws):
        self._method = method
        self._scale_covar = scale_covar
        self._nan_policy = nan_policy
        self._reduce_fcn = reduce_fcn
        self._kws = {}

        if method == 'least_squares':
            self._kws.update({
                'diff_step': 0.001,
                'max_nfev': 5000,
                'ftol': 1e-5,
                'xtol': 1e-5,
                'gtol': 1e-5
            })

    def _fit_impl(self, datasets, model, param_info):

        params_name = param_info.keys()
        params_val = []
        params_min = []
        params_max = []
        for pname, pinfo in param_info.items():
            params_val.append(pinfo['init'])
            params_min.append(pinfo['min'])
            params_max.append(pinfo['max'])

        residuals_size = 0
        for datamap in datasets:
            for key, value in datamap.items():
                residuals_size += value.data().size
        residuals = np.empty(residuals_size)

        residualset = []
        for datamap in datasets:
            residualmap = {}
            for key, value in datamap.items():
                residualmap[key] = np.empty_like(value.data)
            residualset.append(residualmap)

        params = lmfit.parameter.Parameters()

        def parse_parameter(info):
            value = info['init']
            min_ = info.get('min')
            max_ = info.get('max')
            step = info.get('brute_step')
            return value, min_, max_, step

        for key, value in param_info.items():
            value, min_, max_, step = parse_parameter(value)
            key = key.replace('[', '_')
            key = key.replace(']', '_')
            params[key] = lmfit.Parameter(key, value, True, min_, max_, None, step)

        def _iter_cb(params, iter, resid, *fcn_args, **fcn_kws):
            # print("iter: ", iter)
            # print("params: ", params)
            # print("sum: ", np.sum(resid))
            pass

        def residuals_fcn(free_pars, datasets, model, residuals):
            pars = {}
            for i, par_name in enumerate(model.get_param_names()):
                par_name1 = par_name
                par_name1 = par_name1.replace('[', '_')
                par_name1 = par_name1.replace(']', '_')
                pars[par_name] = free_pars[par_name1].value
            print(pars)
            modelset = model.evaluate(pars, False)
            # Generate model data
            residual_offset = 0
            for dataset, model_data in zip(datasets, modelset):
                for name, model in model_data.items():
                    size = model.size
                    data = dataset[name].data()
                    resid = data - model
                    resid[np.isnan(resid)] = 0
                    residuals[residual_offset:residual_offset + size] = resid.ravel()
                    residual_offset += size
            out = np.array(residuals.ravel())
            print("Residual sum: ", np.nansum(out))
            return out

        fcn = residuals_fcn
        fcn_args = (datasets, model, residuals)
        fcn_kws = None
        iter_cb = _iter_cb

        minimizer = lmfit.Minimizer(
            fcn,
            params,
            fcn_args=fcn_args,
            fcn_kws=fcn_kws,
            iter_cb=iter_cb,
            scale_covar=self._scale_covar,
            nan_policy=self._nan_policy,
            reduce_fcn=self._reduce_fcn,
            **self._kws)

        result = minimizer.minimize(method=self._method)

        """
        class Prob:

            def fitness(self, x):
                params = list(zip(params_name, x))
                res = 0
                outputs = model.evaluate(params, False)
                for dataset, output in zip(datasets, outputs):
                    for name in output:
                        dat = dataset[name].data()
                        mdl = output[name]
                        res += np.nansum(np.abs(dat - mdl).astype(np.float64))
                return (res,)

            def get_bounds(self):
                return (params_min, params_max)

            def gradient(self, x):
                return pg.estimate_gradient_h(lambda x: self.fitness(x), x)

        gopt_algos = {
            'gaco': pg.gaco,
            'de': pg.de,
            'sade': pg.sade,
            'de1220': pg.de1220,
            'gwo': pg.gwo,
            'ihs': pg.ihs,
            'pso': pg.pso,
            'pso_gen': pg.pso_gen,
            'sea': pg.sea,
            'sga': pg.sga,
            'simulated_annealing': pg.simulated_annealing,
            'bee_colony': pg.bee_colony,
            'cmaes': pg.cmaes,
            'xnes': pg.xnes,
            'nsga2': pg.nsga2,
            'moead': pg.moead,
            'maco': pg.maco,
            'nspso': pg.nspso
        }

        prob = pg.problem(Prob())

        if self._algo in gopt_algos:
            uda = pg.cmaes(gen=self._gen)
        else:
            uda = pg.nlopt(self._algo)

        algo = pg.algorithm(uda)
        algo.set_verbosity(10)
        pop = pg.population(prob, size=self._size)
        pop.set_x(0, params_val)
        pop = algo.evolve(pop)

        print(f"I used size={self._size} and gen={self._gen}.")
        print("best fit (params): ", pop.champion_x)
        print("best fit (chi2): ", pop.champion_f)
        print("done")
        # #algo.extract(pg.nlopt).ftol_rel = 1e-8
        """
