"""
Bootstrap errors: the spread of the best fits of data made of the model
of a fit plus its residuals, redrawn (see bootstrap).
"""

import abc
import logging
import numbers
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gbkfit.dataset import Dataset
from gbkfit.fitting.base import Fitter, fitter_parser
from gbkfit.fitting.result import Bootstrap
from gbkfit.objective import Objective
from gbkfit.params import ParamEvaluationError, ParamSpace, ParamValues
from gbkfit.utils import iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'BootstrapBlocks',
    'BootstrapOptions',
    'BootstrapScheme',
    'bootstrap',
    'bootstrap_scheme_parser'
]


_log = logging.getLogger(__name__)


class BootstrapScheme(parseutils.TypedSerializable, abc.ABC):
    """
    How a bootstrap redraws the residuals of the data of an observation
    (see bootstrap), keeping the correlations of the noise (e.g. within
    the beam): each kind of data offers the schemes it can (e.g. see
    Dataset.resample_blocks).
    """

    def check(self, dataset: Dataset, params: ParamSpace) -> None:
        """
        Raise ConfigError unless it suits the data of an observation and
        the parameters (by default, it does).
        """

    @abc.abstractmethod
    def resample(
            self,
            dataset: Dataset,
            residuals: Mapping[str, np.ndarray],
            model: Mapping[str, np.ndarray],
            params: ParamSpace,
            values: ParamValues,
            rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """
        Return the residuals of the data of an observation, redrawn.

        Parameters
        ----------
        dataset : Dataset
            The data.
        residuals : Mapping
            The residuals of each data item: NaN where there are none.
        model : Mapping
            The model of each data item, which they are of.
        params : ParamSpace
            The parameters of the fit.
        values : ParamValues
            Their values at the best fit.
        rng : np.random.Generator
            The random numbers.

        Returns
        -------
        dict
            The redrawn residuals of each data item: NaN where it has none.

        Raises
        ------
        ConfigError
            If the data cannot be redrawn by it.
        """


class BootstrapBlocks(BootstrapScheme):
    """
    Residuals redrawn in blocks (see Dataset.resample_blocks): each block
    takes those of a block drawn at random, with replacement, from those
    with residuals in all of their values. Each kind of data says what
    its blocks are (see Dataset.check_blocks; e.g. pixels and channels of
    a cube). Blocks of at least a beam (and of the width of the line
    spread in channels) keep the correlations of the noise.

    Parameters
    ----------
    size : Sequence of float
        The size of the blocks (e.g. x, y and the channels of a cube, in
        pixels and channels).

    Raises
    ------
    ConfigError
        If the size is not of positive numbers.
    """

    @staticmethod
    def type() -> str:
        return 'blocks'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type(), size=list(self._size))

    def __init__(self, size: Sequence[float]):
        size = tuple(size)
        if not size or not all(
                isinstance(n, numbers.Real) and not isinstance(n, bool)
                and n > 0 for n in size):
            raise ConfigError(
                f"the size of the blocks is of positive numbers; it is "
                f"{list(size)}")
        self._size = size

    def size(self) -> tuple[float, ...]:
        """Return the size of the blocks."""
        return self._size

    def check(self, dataset: Dataset, params: ParamSpace) -> None:
        dataset.check_blocks(self._size)

    def resample(
            self,
            dataset: Dataset,
            residuals: Mapping[str, np.ndarray],
            model: Mapping[str, np.ndarray],
            params: ParamSpace,
            values: ParamValues,
            rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        return dataset.resample_blocks(residuals, self._size, rng)


bootstrap_scheme_parser = parseutils.TypedParser(
    BootstrapScheme, [BootstrapBlocks])


class BootstrapOptions(parseutils.Serializable):
    """
    The options of a bootstrap (see bootstrap).

    Parameters
    ----------
    nsamples : int
        The number of refits.
    scheme : BootstrapScheme or Sequence of BootstrapScheme
        How the residuals of the data of every observation are redrawn,
        or of each.
    fitter : Fitter, optional
        The fitter of the refits; by default, that of the fit. The refits
        work out no uncertainty.
    seed : int, optional
        The seed of the random numbers of the redrawn residuals; by
        default, that of the fitter of the refits.

    Raises
    ------
    ConfigError
        If nsamples is below 2.
    """

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'BootstrapOptions':
        info = parseutils.parse_options(
            info, required={'nsamples', 'scheme'},
            optional={'fitter', 'seed'})
        return cls(
            info['nsamples'],
            parseutils.load_option(
                bootstrap_scheme_parser.load, info, 'scheme', required=True),
            parseutils.load_option(fitter_parser.load, info, 'fitter'),
            info.get('seed'))

    def dump(self) -> dict[str, Any]:
        info: dict[str, Any] = dict(
            nsamples=self._nsamples,
            scheme=bootstrap_scheme_parser.dump(self._scheme))
        if self._fitter is not None:
            info['fitter'] = self._fitter.dump()
        if self._seed is not None:
            info['seed'] = self._seed
        return info

    def __init__(
            self,
            nsamples: int,
            scheme: BootstrapScheme | Sequence[BootstrapScheme],
            fitter: Fitter | None = None,
            seed: int | None = None
    ):
        if nsamples < 2:
            raise ConfigError(
                f"a bootstrap needs at least 2 samples; nsamples is "
                f"{nsamples}")
        self._nsamples = nsamples
        self._scheme = list(scheme) if iterutils.is_sequence(scheme) \
            else scheme
        self._fitter = fitter
        self._seed = seed

    def nsamples(self) -> int:
        """Return the number of refits."""
        return self._nsamples

    def fitter(self) -> Fitter | None:
        """Return the fitter of the refits, if given."""
        return self._fitter

    def seed(self) -> int | None:
        """Return the seed of the redrawn residuals, if given."""
        return self._seed

    def schemes(self, nobservations: int) -> list[BootstrapScheme]:
        """
        Return the scheme of each of a number of observations. Raise
        ConfigError unless there is one scheme, or one per observation.
        """
        if not isinstance(self._scheme, list):
            return [self._scheme] * nobservations
        if len(self._scheme) != nobservations:
            raise ConfigError(
                f"the scheme is one, or a list of one for each of the "
                f"{nobservations} observations; it is a list of "
                f"{len(self._scheme)}")
        return self._scheme

    def check(self, objective: Objective, params: ParamSpace) -> None:
        """
        Raise ConfigError unless the options suit the observations of an
        objective and its parameters: their schemes their data (see
        BootstrapScheme.check), and the free properties of the parameters
        the fitter of the refits, if given (see Fitter.load_fit_params).
        """
        schemes = self.schemes(objective.nobservations())
        for i, (scheme, dataset) in enumerate(
                zip(schemes, objective.datasets())):
            index = [i] if isinstance(self._scheme, list) else []
            with parseutils.config_path('scheme', *index):
                scheme.check(dataset, params)
        if self._fitter is not None:
            with parseutils.config_path('fitter'):
                self._fitter.load_fit_params(params)


def bootstrap(
        objective: Objective,
        params: ParamSpace,
        start: Mapping[str, float],
        fitter: Fitter,
        options: BootstrapOptions
) -> Bootstrap:
    """
    Work out the errors of a fit by a bootstrap: refit data made of the
    model of its best fit plus its residuals redrawn, and take the spread
    of the best fits of the refits (see Bootstrap). Unlike the covariance,
    it takes the noise as the residuals have it: correlated (e.g. within
    the beam), and with what the model misses.

    The data of each refit are m + sigma r*, of the model m of the best
    fit (that which the objective compares, of the spectra times their
    amplitudes, if fitted; see Objective), the noise sigma of each value
    (its error, or its fitted noise; see Likelihood.noise), and the
    residuals in units of it, r = (d - m) / sigma, redrawn by the scheme
    of each observation (see BootstrapScheme): a residual moved from a
    noisy value to a quiet one takes the noise of the latter. The values
    without a residual keep their data (the masked values stay masked,
    and those the model lacks keep theirs). Each refit starts at the best
    fit (see ParamSpace.with_values), with the fitter of the options, or
    else that of the fit, which works out no uncertainty. The refits that
    fail (the fitter reports no success or finds no point, or the
    parameters cannot be evaluated) are counted and left out.

    The fitter of the refits may differ from that of the fit (e.g. least
    squares after a sampler or CMA-ES): the bootstrap tells how the best
    fit moves with the noise, which a fitter that starts there tells if
    it finds the nearby minimum. After a sampler, the errors are then
    those of the best fit, not of the median of the posterior; and a
    local fitter keeps to the minimum of the fit.

    Parameters
    ----------
    objective : Objective
        The objective of the fit.
    params : ParamSpace
        Its parameters.
    start : Mapping
        The value of every element at the best fit, by element name (see
        FitSolution.elements).
    fitter : Fitter
        The fitter of the fit.
    options : BootstrapOptions
        The options.

    Returns
    -------
    Bootstrap
        The bootstrap.

    Raises
    ------
    ConfigError
        If the options do not suit the observations or the parameters
        (see BootstrapOptions.check), or the data of an observation cannot
        be redrawn by its scheme (see BootstrapScheme.resample).
    """
    options.check(objective, params)
    refitter = (options.fitter() or fitter).without_uncertainty()
    seed = refitter.seed() if options.seed() is None else options.seed()
    rng = np.random.default_rng(seed)
    start_params = params.with_values(start)
    values = start_params.evaluate()
    group = objective.group()
    schemes = options.schemes(group.nobservations())
    models = _models(objective, values)
    noises = [
        {key: obs.likelihood().noise(key, dataset[key].error(), values)
         for key in dataset}
        for obs, dataset in zip(group.observations(), objective.datasets())]
    residuals = [
        {key: (dataset[key].data() - model[key]) / noise[key]
         for key in dataset}
        for dataset, model, noise in zip(
            objective.datasets(), models, noises)]
    try:
        derived = tuple(params.derived_elements(values))
    except ParamEvaluationError:
        derived = ()
    names = tuple(params.kinds()) + derived
    # (the refits share its whitening of correlated noise; see with_data)
    objective.prepare()
    rows = []
    failures = []
    for k in range(options.nsamples()):
        datasets = []
        for scheme, dataset, residual, model, noise in zip(
                schemes, objective.datasets(), residuals, models, noises):
            drawn = scheme.resample(
                dataset, residual, model, params, values, rng)
            datasets.append(dataset.with_data({
                key: np.where(
                    np.isfinite(drawn[key]),
                    model[key] + noise[key] * drawn[key],
                    dataset[key].data())
                for key in dataset}))
        refit = f"bootstrap refit {k + 1}/{options.nsamples()}"
        try:
            # (closed after the refit: the fitter can keep it alive)
            with objective.with_data(datasets) as refit_objective:
                result = refitter.fit(refit_objective, start_params)
        except (ParamEvaluationError, RuntimeError) as e:
            failures.append(str(e))
            _log.warning(f"{refit} failed: {e}")
            continue
        if not result.success:
            failures.append(f"the fitter did not succeed: {result.message}")
            _log.warning(f"{refit} failed: {failures[-1]}")
            continue
        best = result.best()
        rows.append(
            [best.elements[name] for name in params.kinds()]
            + [best.derived.get(name, np.nan) for name in derived])
        _log.info(f"{refit}: chi2 {best.chi2:.6g}")
    return Bootstrap(
        names=names, samples=np.array(rows, dtype=float).reshape(
            -1, len(names)),
        failures=tuple(failures), options=options.dump())


def _models(
        objective: Objective, values: ParamValues
) -> list[dict[str, np.ndarray]]:
    """
    Return the model of each data item of each observation of an
    objective at the values of the parameters, as the objective compares
    it with the data: of the spectra, times their amplitudes, if fitted.
    """
    # (an evaluation of the objective gives the amplitudes)
    objective.chi2(values)
    amplitudes = objective.amplitudes()
    group = objective.group()
    models = []
    for i, model in enumerate(group.model_h(values)):
        items = {key: item['d'].copy() for key, item in model.items()}
        if amplitudes[i] is not None:
            observation = group.observations()[i]
            key = observation.spectra_item()
            items[key] = items[key] * np.expand_dims(
                amplitudes[i], observation.spectra_axis())
        models.append(items)
    return models
