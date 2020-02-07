
import numpy as np
import pygmo as pg

import gbkfit.fitter


class FitterPygmo(gbkfit.fitter.Fitter):

    @staticmethod
    def type():
        return 'pygmo'

    @classmethod
    def load(cls, info):
        algo = info['algo']
        size = info.get('size')
        gen = info.get('gen')
        return cls(algo, size, gen)

    def dump(self):
        return {'type': self.type()}

    def __init__(self, algo, size, gen):
        if size is None:
            size = 1
        if gen is None:
            gen = 50
        self._algo = algo
        self._size = size
        self._gen = gen

    def _fit_impl(self, datasets, model, param_info):

        params_name = param_info.keys()
        params_val = []
        params_min = []
        params_max = []
        for pname, pinfo in param_info.items():
            params_val.append(pinfo['init'])
            params_min.append(pinfo['min'])
            params_max.append(pinfo['max'])

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
