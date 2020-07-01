
import astropy.io.fits as fits
import numpy as np

import gbkfit.params.interpreter


def _dump_posterior(params, samples, loglikes, logprobs, prefix=''):
    combined = np.column_stack((logprobs, loglikes, samples))
    width1 = 26
    width2 = 18
    fmt = f'%{width1}.{width2}e'
    samples_file = f'{prefix}posterior_samples.txt'
    loglikes_file = f'{prefix}posterior_loglikes.txt'
    logprobs_file = f'{prefix}posterior_logprobs.txt'
    combined_file = f'{prefix}posterior_combined.txt'
    samples_hdr = ''.join(param.rjust(width1) for param in params)
    loglikes_hdr = 'log_likelihood'.rjust(width1)
    logprobs_hdr = 'log_probability'.rjust(width1)
    combined_hdr = logprobs + loglikes_hdr + samples_hdr
    np.savetxt(samples_file, samples, fmt=fmt, header=samples_hdr)
    np.savetxt(loglikes_file, loglikes, fmt=fmt, header=loglikes_hdr)
    np.savetxt(logprobs_file, logprobs, fmt=fmt, header=logprobs_hdr)
    np.savetxt(combined_file, combined, fmt=fmt, header=combined_hdr)
    return dict(
        samples=samples_file,
        loglikes=loglikes_file,
        logprobs=logprobs_file,
        combined=combined_file)


class FitterResultSolution:

    def __init__(
            self, samples, loglikes, logprobs,
            best, mean, covar, stddev, chisqr, rchisqr,
            model, residual):
        self._samples = samples
        self._loglikes = loglikes
        self._logprobs = logprobs
        self._best = best
        self._mean = mean
        self._covar = covar
        self._stddev = stddev
        self._chisqr = chisqr
        self._rchisqr = rchisqr
        self._model = model
        self._residual = residual

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
    def best(self):
        return self._best

    @property
    def mean(self):
        return self._mean

    @property
    def covar(self):
        return self._covar

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
    def model(self):
        return self._model

    @property
    def residual(self):
        return self._residual


class FitterResult:

    @classmethod
    def load(cls, info):
        return cls()

    def dump(self):

        objective = self.objective()
        info = dict(
            drivers=[driver.dump() for driver in objective.drivers()],
            datasets=[dataset.dump() for dataset in objective.datasets()],
            dmodels=[dmodel.dump() for dmodel in objective.dmodels()],
            gmodels=[gmodel.dump() for gmodel in objective.gmodels()],
            modes=[])

        for i, sol in enumerate(self.solutions()):

            prefix = f'solution{i}_'

            if sol.best is not None:
                foo = dict(zip([], sol.best))
            if sol.mean is not None:
                pass
            if sol.covar is not None:
                pass
            if sol.stddev is not None:
                pass

            if sol.samples and sol.loglikes and sol.logprobs:
                info.update(posterior=_dump_posterior(
                    params, sol.samples, sol.loglikes, sol.logprobs, prefix))

            for j, model in enumerate(sol.model):
                for k, v in model.items():
                    file = f'{prefix}mdl_{j}_{k}'
                    fits.writeto(file, v)

    def __init__(self, objective, params, extra=None):
        self._objective = objective
        self._params = params
        self._solutions = []
        self._interpreter = gbkfit.params.interpreter.ParamInterpreter(
            params.descs(), params.exprs())

    def objective(self):
        return self._objective

    def params(self):
        return self._params

    def solutions(self):
        return tuple(self._solutions)

    def params_dict(self, mode=None):
        if mode is None:
            mode = self.champion()
        eparams = dict(zip(self.params().enames_free(), mode.best()))
        return self._interpreter.evaluate(eparams)

    def add_mode(
            self, samples=None, loglikes=None, logprobs=None,
            mode=None, mean=None, covar=None, stddev=None):
        has_posterior = samples and loglikes and logprobs
        if not has_posterior and mode is None:
            raise RuntimeError()
        if has_posterior:
            if mode is None:
                mode = samples[np.argmax(loglikes)]
            if mean is None:
                mean = np.mean(samples, axis=0)
            if covar is None:
                covar = None
        if covar:
            if stddev is None:
                stddev = None
        params = self.params_dict(mode)
        model = self.objective().model(params)
        residual = self.objective().residual_nddata(params)
        chisqr = None
        rchisqr = None
        self._solutions.append(
            FitterResultSolution(
                samples, loglikes, logprobs,
                mode, mean, covar, stddev, chisqr, rchisqr,
                model, residual))
