
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
    def __init__(self):
        pass
