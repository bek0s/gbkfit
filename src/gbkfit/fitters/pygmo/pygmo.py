
import abc
import logging

import numpy as np
import pygmo as pg

import gbkfit.fitter


log = logging.getLogger(__name__)


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

    def _impl_fit(self, data, model, params):

        pvals = []
        pmins = []
        pmaxs = []

        # Prepare parameter properties
        for pname in model.get_param_names():
            pinfo = params[pname]
            pvals.append(pinfo['init'])
            pmins.append(pinfo.get('min', -np.nan))
            pmaxs.append(pinfo.get('max', +np.nan))

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

        class Problem:

            def fitness(self, pvalues):
                params = dict(zip(model.get_param_names(), pvalues))
                log.info(params)
                outputs = model.evaluate(params, False)
                tres = 0
                for dataset, output in zip(data, outputs):
                    for name in output:
                        dat = dataset[name].data()
                        msk = dataset[name].mask()
                        err = dataset[name].error()
                        mdl = output[name]
                        res = msk * (mdl - dat) / err
                        #res[np.isnan(res)] = 0

                        """
                        msk_dat = msk
                        msk_mdl = np.isfinite(mdl).astype(np.int)
                        import astropy.io.fits as fits
                        mm = (msk_dat + msk_mdl) * msk
                        fits.writeto(f'mask_{name}.fits', msk, overwrite=True)
                        fits.writeto(f'test_{name}.fits', mm, overwrite=True)
                        """
                        """
                        if name != 'mmap0':
                            res *= 0.1
                        """

                        tres += np.nansum(np.abs(res))

                #tres = np.abs(tres)
                print("Residual sum: ", tres)
                return tres,

            def get_bounds(self):
                return pmins, pmaxs

            def gradient(self, x):
                return pg.estimate_gradient_h(lambda x: self.fitness(x), x)

        pp = Problem()
        prob = pg.problem(pp)

        if self._algo in gopt_algos:
            uda = pg.cmaes(gen=250, force_bounds=True, sigma0=1)
        else:
            uda = pg.nlopt(self._algo)

        algo = pg.algorithm(uda)
        algo.set_verbosity(10)
        pop = pg.population(prob, size=self._size)
        pop.set_x(0, pvals)
        pop.set_x(1, pvals)
        pop = algo.evolve(pop)

        print("------------------------------")
        print("Results")
        print("------------------------------")
        print(f"size={self._size} and gen={self._gen}")
        for key, value in zip(params.keys(), pop.champion_x):
            print(f"{key:<{6}}: {round(value, 3):<{6}}")
        print("best fit (params): ", pop.champion_x)
        print("best fit (chi2): ", pop.champion_f)
        print("done")
        # #algo.extract(pg.nlopt).ftol_rel = 1e-8

        final_params = dict(zip(params.keys(), pop.champion_x))
        outputs = model.evaluate(final_params, explode_params=False)

        import astropy.io.fits as fits
        for dataset, output in zip(data, outputs):
            for name in output.keys():
                dat = dataset[name].data()
                mdl = output[name]
                residual = mdl - dat
                fits.writeto(f'result_data_{name}.fits', dat, overwrite=True)
                fits.writeto(f'result_model_{name}.fits', mdl, overwrite=True)
                fits.writeto(f'result_residual_{name}.fits', residual, overwrite=True)
