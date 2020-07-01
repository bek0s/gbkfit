
import abc

import astropy.io.fits as fits
import numpy as np

from gbkfit.utils.parsers import parser_functions as _parser_functions


class FitterResultMode:

    def __init__(
            self, best, mean, stddev, chisqr, rchisqr, samples, loglikes,
            logprobs, models, residuals):
        self._best = best
        self._mean = mean
        self._stddev = stddev
        self._chisqr = chisqr
        self._rchisqr = rchisqr
        self._samples = samples
        self._loglikes = loglikes
        self._logprobs = logprobs
        self._models = models
        self._residuals = residuals

    @property
    def best(self):
        return self._best

    @property
    def mean(self):
        return self._mean

    @property
    def stddev(self):
        return self._stddev

    @property
    def chisqr(self):
        return self._chisqr

    @property
    def rchisqr(self):
        return self._rchisqr

    @property
    def samples(self):
        return self._samples

    @property
    def loglikes(self):
        return self._loglikes

    @property
    def logprobs(self):
        return self._logprobs

    @property
    def models(self):
        return self._models

    @property
    def residuals(self):
        return self._residuals


class FitterResult:

    @classmethod
    def load(cls, info):

        import gbkfit.dataset
        import gbkfit.driver
        import gbkfit.dmodel
        import gbkfit.gmodel
        import gbkfit.params
        import gbkfit.model
        import gbkfit.parameters

        def _prepare_params(params):
            param_infos = {}
            param_exprs = {}
            for key, value in params.items():
                if isinstance(value, dict):
                    param_infos[key] = value
                else:
                    param_exprs[key] = value
            return param_infos, param_exprs

        datasets = gbkfit.dataset.create(info['datasets'])
        #models = gbkfit.model.parse(info['models'], datasets)

        gmodels = [gbkfit.gmodel.parser.parse(item) for item in info['gmodels']]
        drivers = [gbkfit.driver.parser.parse(item) for item in info['drivers']]
        dmodels = [gbkfit.dmodel.parser.parse(item) for item in info['dmodels']]
        models = gbkfit.model.Model(drivers, dmodels, gmodels)
        #parset = gbkfit.parameters.parse_parset(info['params'])
        parset, exprs = _prepare_params(info['params'])
        models.set_param_fixed(exprs)

        #print()

        result = FitterResult(datasets, models, parset)

        for mode in info['modes']:
            post_info = mode['posterior']
            if not post_info:
                best = mode['best'].values() if 'best' in mode else None
                mean = mode['mean'].values() if 'mean' in mode else None
                stddev = mode['stddev'].values() if 'stddev' in mode else None
                result.add_mode_from_bestfit(best, mean, stddev)
            else:
                samples = np.genfromtxt(post_info['samples'], skip_header=1)
                loglikes = np.genfromtxt(post_info['loglikes'], skip_header=1)
                logprobs = np.genfromtxt(post_info['logprobs'], skip_header=1)
                result.add_mode_from_samples(samples, loglikes, logprobs)

        return result


    def dump(self):

        info = {
            'datasets': self.datasets.dump(),
            'drivers': [driver.dump() for driver in self.models.drivers()],
            'dmodels': [dmodel.dump() for dmodel in self.models.dmodels()],
            'gmodels': [gmodel.dump() for gmodel in self.models.gmodels()],
            #'parset': self.parset.dump(self.models.params),
            'modes': []
        }

        for i, mode in enumerate(self.modes):

            info['modes'].append({})
            mode_info = info['modes'][i]

            if mode.best is not None:

                mode_info['best'] = dict(
                    zip(self._models.get_param_names(), mode.best))

            if mode.mean is not None:

                mode_info['mean'] = dict(
                    zip(self._models.get_param_names(), mode.mean))

            if mode.stddev is not None:

                mode_info['stddev'] = dict(
                    zip(self._models.get_param_names(), mode.stddev))

            mode_info['chisqr'] = mode.chisqr
            mode_info['rchisqr'] = mode.rchisqr

            mode_info['posterior'] = None

            if mode.samples is not None:

                mode_info['posterior'] = {}
                posterior_info = mode_info['posterior']

                cwidth = 26
                dwidth = 18

                samples_file = 'mode_{}_posterior_samples.txt'.format(i)
                loglikes_file = 'mode_{}_posterior_loglikes.txt'.format(i)
                logprobs_file = 'mode_{}_posterior_logprobs.txt'.format(i)
                combined_file = 'mode_{}_posterior_combined.txt'.format(i)

                samples_fmt = '%{}.{}e'.format(cwidth, dwidth)
                loglikes_fmt = '%{}.{}e'.format(cwidth, dwidth)
                logprobs_fmt = '%{}.{}e'.format(cwidth, dwidth)
                combined_fmt = '%{}.{}e'.format(cwidth, dwidth)

                #samples_hdr = ''.join('{:>{}}'.format(par, cwidth) for par in self.parset.pars_free.keys())
                samples_hdr = ''.join('{:>{}}'.format(par, cwidth) for par in self._models.get_param_names())
                loglikes_hdr = '{:>{}}'.format('log_likelihood', cwidth)
                logprobs_hdr = '{:>{}}'.format('log_probability', cwidth)
                combined_hdr = logprobs_hdr + loglikes_hdr + samples_hdr

                combined_data = np.column_stack((
                    mode.logprobs,
                    mode.loglikes,
                    mode.samples))

                np.savetxt(samples_file, mode.samples, fmt=samples_fmt,
                           header=samples_hdr, delimiter='', comments='')
                np.savetxt(loglikes_file, mode.loglikes, fmt=loglikes_fmt,
                           header=loglikes_hdr, delimiter='', comments='')
                np.savetxt(logprobs_file, mode.logprobs, fmt=logprobs_fmt,
                           header=logprobs_hdr, delimiter='', comments='')
                np.savetxt(combined_file, combined_data, fmt=combined_fmt,
                           header=combined_hdr, delimiter='', comments='')

                posterior_info['samples'] = samples_file
                posterior_info['loglikes'] = loglikes_file
                posterior_info['logprobs'] = logprobs_file
                posterior_info['combined'] = combined_file

            mode_info['models'] = []
            for j, model in enumerate(mode.models):
                mode_info['models'].append({})
                for key, value in model.items():
                    file = 'mode_{}_mdl_{}_{}.fits'.format(i, j, key)
                    fits.writeto(file, value, overwrite=True)
                    mode_info['models'][j][key] = file

            mode_info['residuals'] = []
            for j, residual in enumerate(mode.residuals):
                mode_info['residuals'].append({})
                for key, value in residual.items():
                    file = 'mode_{}_res_{}_{}.fits'.format(i, j, key)
                    fits.writeto(file, value, overwrite=True)
                    mode_info['residuals'][j][key] = file

        return info

    def __init__(self, datasets, models, parset):
        self._datasets = datasets
        self._models = models
        self._parset = parset
        self._pars_all = parset.keys()
        self._pars_free = models.get_param_names(free=True, fixed=False)  # parset.pars_free.keys()
        self._pars_fixed = models.get_param_names(free=False, fixed=True)  # parset.pars_fixed
        self._ncall = None
        self._dof = 0
        for dataset in self.datasets:
            for data in dataset.values():
                self._dof += np.count_nonzero(data.mask)
        self._dof -= len(self._pars_free) - 1
        self._modes = []

    @property
    def datasets(self):
        return self._datasets

    @property
    def models(self):
        return self._models

    @property
    def parset(self):
        return self._parset

    @property
    def dof(self):
        return self._dof

    @property
    def nmodes(self):
        return len(self._modes)

    @property
    def modes(self):
        return self._modes

    def add_mode_from_bestfit(self, best, mean, stddev):
        self._add_mode(best, mean, stddev, None, None, None)

    def add_mode_from_samples(self, samples, loglikes, logprobs):
        best = samples[np.argmax(loglikes)]
        mean = np.mean(samples, axis=0)
        stddev = np.std(samples, axis=0)
        self._add_mode(best, mean, stddev, samples, loglikes, logprobs)

    def _add_mode(self, best, mean, stddev, samples, loglikes, logprobs):
        #pars = dict(zip(self._parset.pars_free.keys(), best))
        pars = dict(zip(self._models.get_param_names(), best))
        #pars.update(self._parset.pars_fixed)
        models = self._models.evaluate(pars, False)
        #print(pars)
        #raise RuntimeError()
        #exit(0)
        chisqr = 0
        residuals = []
        for i, (dataset, modelset) in enumerate(zip(self.datasets, models)):
            residuals.append({})
            for key, model in modelset.items():
                data = dataset[key].data
                mask = dataset[key].mask
                error = dataset[key].error
                residual = model - data
                chisqr += np.nansum(mask * (residual * residual / error))
                residuals[i][key] = residual
        rchisqr = chisqr / self.dof
        self._modes.append(FitterResultMode(best, mean, stddev, chisqr,
                                            rchisqr, samples, loglikes,
                                            logprobs, models, residuals))


class Fitter(abc.ABC):

    @abc.abstractmethod
    def fit(self, dataset, model, parset):
        pass


register_parser, parse = _parser_functions(Fitter)