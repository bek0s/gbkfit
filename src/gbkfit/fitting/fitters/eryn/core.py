"""
The Eryn MCMC sampler, with reversible jumps.
"""

import dataclasses
import typing
from collections.abc import Mapping, Sequence
from typing import Any

import eryn.ensemble
import eryn.moves
import eryn.prior
import eryn.state
import numpy as np
import numpy.typing as npt

from gbkfit.fitting.base import FitRun, Fitter
from gbkfit.fitting.control import FitStopped
from gbkfit.fitting.fitparams import FitParam, FitParamClass, FitParamWalkers
from gbkfit.fitting.fitters._detail import start_walkers
from gbkfit.fitting.priors import Prior
from gbkfit.fitting.problem import FitProblem
from gbkfit.fitting.result import Evidence
from gbkfit.fitting.uncertainty import UncertaintyOptions
from gbkfit.params import ParamSpace, param_tables
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'FitParamEryn',
    'FitterEryn'
]


# The log-likelihood that Eryn gets for points that cannot be evaluated:
# finite, as Eryn asks (an infinite one breaks its acceptance)
_IMPOSSIBLE = -1e300

# The name of the branch of the free elements that are always there
_FIXED = 'fixed'

# The steps between two adaptations of the proposals during the burn-in,
# and the acceptance they aim at (that of a Gaussian random walk in a few
# dimensions; Roberts et al. 1997)
_ADAPTATION_INTERVAL = 25
_ADAPTATION_TARGET = 0.25


class FitParamEryn(FitParamWalkers):
    """
    The options of a free element for Eryn: its prior, and where the
    walkers start (see FitParamWalkers).
    """


class _ErynPrior:
    """A prior as Eryn takes it: its log density, and draws from it."""

    def __init__(self, prior: Prior, rng: np.random.Generator):
        self._prior = prior
        self._rng = rng

    def logpdf(self, x: npt.ArrayLike) -> np.ndarray:
        return self._prior.log_pdf(x)

    def rvs(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        return self._prior.ppf(self._rng.uniform(size=size))


@dataclasses.dataclass
class _Table:
    """
    A table of parameters whose rows Eryn varies (see ParamTable): its
    name, the free elements of each row it varies (rows, columns: their
    indices in the free values), the values of an unused row (the empty
    row in its columns, the median of the prior in the others), the
    least rows it uses of them, and the names of its columns.
    """
    name: str
    elements: np.ndarray
    unused: np.ndarray
    min_rows: int
    columns: tuple[str, ...]


@dataclasses.dataclass
class _Layout:
    """
    How the branches of Eryn make the free values: the elements of the
    branch of the elements that are always there (their indices), and the
    tables whose rows Eryn varies.
    """
    fixed: np.ndarray
    tables: list[_Table]

    def branches(self) -> list[str]:
        """Return the names of the branches, in order."""
        return [_FIXED] + [table.name for table in self.tables]

    def points(
            self,
            coords: Sequence[np.ndarray | None],
            groups: Sequence[np.ndarray | None],
            ndim: int
    ) -> np.ndarray:
        """
        Return the free values of the walkers of a call of Eryn (one per
        row): the coordinates of the active leaves of each branch, and
        the walker of each (see EnsembleSampler with provide_groups). The
        leaves of a table go to its first rows, in order; its other rows
        get its unused values.
        """
        nwalkers = len(coords[0])
        points = np.empty((nwalkers, ndim))
        points[groups[0][:, None], self.fixed] = coords[0]
        for table, leaves, walkers in zip(self.tables, coords[1:], groups[1:]):
            points[:, table.elements] = table.unused
            if leaves is None or not len(leaves):
                continue
            # (the rank of each leaf among those of its walker)
            order = np.argsort(walkers, kind='stable')
            ranks = np.empty(len(walkers), dtype=int)
            first = np.searchsorted(walkers[order], walkers[order])
            ranks[order] = np.arange(len(walkers)) - first
            points[walkers[:, None], table.elements[ranks]] = leaves
        return points

    def samples(
            self,
            fixed: np.ndarray,
            leaves: list[np.ndarray],
            active: list[np.ndarray],
            ndim: int
    ) -> np.ndarray:
        """
        Return the free values of samples of the chain (one per row):
        those of the fixed branch (samples, elements), and the leaves of
        each table (samples, leaves, columns) with which are active; NaN
        in the rows of a table a sample does not use.
        """
        samples = np.full((len(fixed), ndim), np.nan)
        samples[:, self.fixed] = fixed
        for table, coords, used in zip(self.tables, leaves, active):
            for i in range(len(samples)):
                rows = coords[i][used[i]]
                samples[i, table.elements[:len(rows)]] = rows
        return samples


class FitterEryn(Fitter):
    """
    MCMC sampling with Eryn (Karnesis et al. 2023): an ensemble of walkers
    at several temperatures (parallel tempering), with reversible jumps
    that vary the number of rows of the tables of parameters (see
    ParamTable): e.g. how many blobs a mixture trait has. Other fitters
    fit every row; Eryn uses from min_rows of them (0 by default) to all,
    adding and removing rows (new ones drawn from the priors), and gives
    the posterior of their number. The rows it does not use get the empty
    row of their table (e.g. a blob of amplitude 0).

    Only the rows of a table that share all their settings are
    interchangeable, and only they can be dropped: a row with settings of
    its own (see ParamSpace.own_settings: e.g. 'pt1_r[0]') is always used,
    and so are the rows whose empty columns are not free (e.g. blobs of a
    fixed amplitude), and all those of a table of a parameter with a
    mode.

    The free elements take a prior, and optionally where the walkers
    start (see FitParamEryn). The walkers are evaluated in batches. It
    gives the posterior (the coldest chain after the burn-in), with NaN
    in the rows a sample does not use; the number of rows of each table
    in each sample (diagnostics 'rows'); and the evidence, by stepping
    stones (the hottest chain at infinite temperature), without priors on
    several parameters (with them, the evidence is not given). Its best
    solution is the point with the highest likelihood it evaluated.

    Parameters
    ----------
    nwalkers : int, optional
        The number of walkers at each temperature (by default, the larger
        of 32 and twice the number of free elements).
    ntemps : int, optional
        The number of temperatures.
    nsteps : int, optional
        The number of steps of each walker.
    burnin : int, optional
        The steps discarded at the start (by default, half), during which
        the temperatures and the proposals adapt.
    thin : int, optional
        Keep every thin-th step of the chain.
    min_rows : Mapping[str, int], optional
        The least rows that Eryn uses of tables, by the name of the table:
        that of its parameters without their column (e.g. 'pt1' of
        'pt1_r'); 0 by default.
    seed : int, optional
        The seed of the random numbers.
    uncertainty : UncertaintyOptions, optional
        How to work out the uncertainty of the best solution.

    Raises
    ------
    ConfigError
        If nwalkers, ntemps, nsteps or thin is not positive, burnin is not
        in [0, nsteps), thin keeps no step after the burn-in, or a value
        of min_rows is negative.
    """

    @staticmethod
    def type() -> str:
        return 'eryn'

    @staticmethod
    def fit_param_class() -> FitParamClass:
        return FitParamEryn

    def libraries(self) -> tuple[str, ...]:
        return ('eryn',)

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), nwalkers=self._nwalkers, ntemps=self._ntemps,
            nsteps=self._nsteps, burnin=self._burnin, thin=self._thin,
            min_rows=self._min_rows) | self.dump_common()

    def __init__(
            self,
            nwalkers: int | None = None,
            ntemps: int = 8,
            nsteps: int = 5000,
            burnin: int | None = None,
            thin: int = 1,
            min_rows: Mapping[str, int] | None = None,
            seed: int = 0,
            uncertainty: UncertaintyOptions | None = None
    ):
        super().__init__(seed, uncertainty)
        for name, value in dict(
                nwalkers=nwalkers, ntemps=ntemps, nsteps=nsteps,
                thin=thin).items():
            if value is not None and not value > 0:
                raise ConfigError(
                    f"option '{name}' must be positive; it is {value}")
        if burnin is not None and not 0 <= burnin < nsteps:
            raise ConfigError(
                f"option 'burnin' must be in [0, nsteps); it is {burnin}")
        kept = nsteps - (nsteps // 2 if burnin is None else burnin)
        if thin > kept:
            raise ConfigError(
                f"option 'thin' ({thin}) keeps no step of the {kept} after "
                f"the burn-in")
        min_rows = dict(min_rows or {})
        if negative := [k for k, v in min_rows.items() if not v >= 0]:
            raise ConfigError(
                f"option 'min_rows' must be at least 0; it is not for "
                f"{negative}")
        self._nwalkers = nwalkers
        self._ntemps = ntemps
        self._nsteps = nsteps
        self._burnin = burnin
        self._thin = thin
        self._min_rows = min_rows

    def _run(
            self,
            problem: FitProblem,
            fit_params: list[FitParam],
            rng: np.random.Generator
    ) -> FitRun:
        params = typing.cast(list[FitParamEryn], fit_params)
        ndim = problem.ndim()
        layout = _layout(problem.params(), problem.names(), params,
                         self._min_rows)
        branches = layout.branches()
        nwalkers = self._nwalkers or max(32, 2 * ndim)
        burnin = self._nsteps // 2 if self._burnin is None else self._burnin
        priors = {
            _FIXED: {i: _ErynPrior(params[j].prior(), rng)
                     for i, j in enumerate(layout.fixed)}}
        for table in layout.tables:
            priors[table.name] = {
                i: _ErynPrior(params[j].prior(), rng)
                for i, j in enumerate(table.elements[0])}

        def log_like(coords: Any, groups: Any) -> np.ndarray:
            # (Eryn gives the arrays of a single branch out of their list)
            if not layout.tables:
                coords, groups = [coords], [groups]
            points = layout.points(coords, groups, ndim)
            values = problem.nested_log_likelihood_batch(points)
            return np.where(np.isfinite(values), values, _IMPOSSIBLE)

        proposals = _proposals(layout, params)
        sampler = eryn.ensemble.EnsembleSampler(
            nwalkers,
            {_FIXED: len(layout.fixed)} | {
                t.name: len(t.columns) for t in layout.tables},
            log_like, priors,
            branch_names=branches,
            nleaves_max={_FIXED: 1} | {
                t.name: len(t.elements) for t in layout.tables},
            nleaves_min={_FIXED: 1} | {
                t.name: t.min_rows for t in layout.tables},
            tempering_kwargs=dict(
                ntemps=self._ntemps, Tmax=np.inf, stop_adaptation=burnin),
            moves=[
                eryn.moves.GaussianMove({name: cov}, gibbs_sampling_setup=name)
                for name, cov in proposals.items()],
            rj_moves=True if layout.tables else None,
            update_fn=_Adaptation(proposals, burnin),
            update_iterations=_ADAPTATION_INTERVAL,
            vectorize=True, provide_groups=True)
        state = _start(problem, params, layout, nwalkers, self._ntemps, rng)
        # A fit asked to stop keeps the chain so far (half of it burn-in,
        # if shorter than the burn-in)
        stopped = False
        try:
            sampler.run_mcmc(state, self._nsteps, thin_by=1)
        except FitStopped:
            stopped = True
        nsteps = sampler.iteration
        burnin = min(burnin, nsteps // 2)
        if nsteps - burnin < 1:
            raise FitStopped("eryn was stopped before its first steps")
        chain = sampler.get_chain(
            discard=burnin, thin=self._thin, branch_names=branches,
            temp_index=0)
        inds = sampler.get_inds(
            discard=burnin, thin=self._thin, branch_names=branches,
            temp_index=0)
        flat = {b: c.reshape(-1, *c.shape[-2:]) for b, c in chain.items()}
        used = {b: i.reshape(-1, i.shape[-1]) for b, i in inds.items()}
        samples = layout.samples(
            flat[_FIXED][:, 0], [flat[t.name] for t in layout.tables],
            [used[t.name] for t in layout.tables], ndim)
        best = problem.best()
        if best is None:
            raise RuntimeError("eryn found no point with a finite chi2")
        evidence = None
        if not stopped and not problem.has_joint_priors():
            log_z, error = sampler.backend.get_evidence_estimate(
                discard=burnin, thin=self._thin, return_error=True,
                method='ss')
            evidence = Evidence(float(log_z), float(error))
        rows = {
            t.name: np.bincount(
                used[t.name].sum(axis=1),
                minlength=len(t.elements) + 1).tolist()
            for t in layout.tables}
        message = f'stopped after {nsteps} of {self._nsteps} steps' \
            if stopped else f'ran {nsteps} steps'
        return FitRun(
            solutions=[best[0]],
            success=not stopped,
            message=message,
            samples=samples,
            evidence=evidence,
            diagnostics=dict(
                rows=rows, nsteps=nsteps,
                acceptance_fraction=float(np.mean(
                    sampler.acceptance_fraction))),
            raw=sampler)


def _layout(
        params: ParamSpace,
        names: Sequence[str],
        fit_params: Sequence[FitParamEryn],
        min_rows: Mapping[str, int]
) -> _Layout:
    """
    Return how the branches of Eryn make the free values (see _Layout):
    the tables of parameters whose rows it can drop (see FitterEryn), of
    the free elements of those rows, and the other free elements.

    Raises
    ------
    ConfigError
        If min_rows names no table whose rows Eryn varies, or asks for
        more rows than a table has.
    """
    index = {name: i for i, name in enumerate(names)}
    own = params.own_settings()
    tables = []
    for table, columns in param_tables(params.pdescs()):
        if any(name in params.modes() for name in columns.values()):
            continue
        size = params.pdescs()[next(iter(columns.values()))].size()
        empty = table.empty()
        # (as ParamSpace names the elements)
        cells = {c: [f'{name}[{i}]' for i in range(size)]
                 for c, name in columns.items()}
        free = tuple(c for c in columns if cells[c][0] in index)
        varied = [
            i for i in range(size)
            if not any(cells[c][i] in own for c in columns)
            and all(cells[c][i] in index for c in empty)
            and tuple(c for c in columns if cells[c][i] in index) == free]
        if not varied or not free:
            continue
        name = next(iter(columns.values()))[:-len(next(iter(columns))) - 1]
        least = min_rows.get(name, 0)
        if least > size:
            raise ConfigError(
                f"option 'min_rows' asks for {least} rows of table "
                f"'{name}', which has {size}")
        elements = np.array(
            [[index[cells[c][i]] for c in free] for i in varied])
        unused = np.array([
            empty[c] if c in empty
            else float(fit_params[elements[0, j]].prior().ppf(0.5))
            for j, c in enumerate(free)])
        tables.append(_Table(
            name, elements, unused, max(0, least - (size - len(varied))),
            free))
    if unknown := set(min_rows) - {t.name for t in tables}:
        raise ConfigError(
            f"option 'min_rows' names tables whose rows Eryn does not "
            f"vary: {sorted(unknown)}; it varies "
            f"{[t.name for t in tables]}")
    in_tables = {int(i) for t in tables for i in t.elements.ravel()}
    fixed = np.array([i for i in range(len(names)) if i not in in_tables])
    return _Layout(fixed, tables)


def _proposals(
        layout: _Layout,
        fit_params: Sequence[FitParamEryn]
) -> dict[str, np.ndarray]:
    """
    Return the covariance of the Gaussian proposals of the walkers within
    a number of rows, by branch: of steps of a tenth of the scale of each
    element (see FitParam.scale).
    """
    def covariance(elements: Sequence[int]) -> np.ndarray:
        return np.diag([(0.1 * fit_params[i].scale()) ** 2 for i in elements])
    return {_FIXED: covariance(layout.fixed)} | {
        t.name: covariance(t.elements[0]) for t in layout.tables}


class _Adaptation:
    """
    The adaptation of the Gaussian proposals of the walkers within a
    number of rows (one move per branch), every few steps of the burn-in
    (an update function of Eryn): each scaled by its acceptance in the
    coldest chain since the last adaptation, towards _ADAPTATION_TARGET;
    that of the fixed branch shaped by the covariance of the coldest
    walkers too (2.38^2 / d of it; the leaves of a table are told apart by
    no label, so theirs keeps the shape it starts with).
    """

    def __init__(self, proposals: dict[str, np.ndarray], burnin: int):
        self._covariances = dict(proposals)
        self._scales = dict.fromkeys(proposals, 1.0)
        self._accepted: dict[str, float] = {}
        self._burnin = burnin

    def __call__(
            self,
            step: int,
            state: eryn.state.State,
            sampler: eryn.ensemble.EnsembleSampler
    ) -> None:
        if step >= self._burnin:
            return
        coords = state.branches[_FIXED].coords[0, :, 0, :]
        if coords.shape[1] and len(coords) > coords.shape[1]:
            spread = np.atleast_2d(np.cov(coords, rowvar=False))
            self._covariances[_FIXED] = 2.38 ** 2 / coords.shape[1] * (
                spread + 1e-12 * np.diag(np.diag(spread) + 1))
        for move, name in zip(sampler.moves, self._covariances):
            accepted = float(np.mean(move.accepted[0]))
            rate = (accepted - self._accepted.get(name, 0.0)) \
                / _ADAPTATION_INTERVAL
            self._accepted[name] = accepted
            self._scales[name] *= np.exp(2 * (rate - _ADAPTATION_TARGET))
            move.all_proposal[name] = eryn.moves.GaussianMove(
                {name: self._scales[name] ** 2 * self._covariances[name]}
            ).all_proposal[name]


def _start(
        problem: FitProblem,
        params: Sequence[FitParamEryn],
        layout: _Layout,
        nwalkers: int,
        ntemps: int,
        rng: np.random.Generator
) -> eryn.state.State:
    """
    Return the starting state: the walkers of every temperature at points
    of finite posterior density (see start_walkers), using the least rows
    of the tables (births then add the rows the data need, rather than
    deaths remove those split across their blobs).
    """
    coords, _ = start_walkers(problem, list(params), nwalkers, rng)
    branches = {_FIXED: coords[:, layout.fixed][:, None, :]}
    for table in layout.tables:
        branches[table.name] = coords[:, table.elements]
    state = {
        name: np.broadcast_to(values, (ntemps, *values.shape)).copy()
        for name, values in branches.items()}
    inds = {_FIXED: np.ones(state[_FIXED].shape[:-1], dtype=bool)}
    for table in layout.tables:
        used = np.zeros(state[table.name].shape[:-1], dtype=bool)
        used[..., :table.min_rows] = True
        inds[table.name] = used
    return eryn.state.State(state, inds=inds)
