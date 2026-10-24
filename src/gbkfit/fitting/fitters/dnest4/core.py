"""
The DNest4 diffusive nested sampler, which also varies the number of
rows of tables of parameters.
"""

import math
import tempfile
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from gbkfit.fitting.base import FitRun, Fitter
from gbkfit.fitting.control import FitStopped
from gbkfit.fitting.fitparams import FitParam, FitParamClass, FitParamPrior
from gbkfit.fitting.fitters._tables import TableLayout, table_layout
from gbkfit.fitting.problem import FitProblem
from gbkfit.fitting.result import Evidence
from gbkfit.fitting.uncertainty import UncertaintyOptions
from gbkfit.utils.parseutils import ConfigError

from . import _dnest4
from ._output import DNest4Files, DNest4Output, posterior


__all__ = [
    'FitterDNest4'
]


# The log-likelihood that DNest4 gets for points that cannot be
# evaluated: finite, as DNest4 sums and compares them
_IMPOSSIBLE = -1e300

# The least time between two reports of the progress (s)
_PROGRESS_INTERVAL = 10.0


class FitterDNest4(Fitter):
    """
    Diffusive nested sampling with DNest4 (Brewer et al. 2011; Brewer &
    Foreman-Mackey 2018), which also varies the number of rows of the
    tables of parameters (see ParamTable): e.g. how many blobs a mixture
    trait has. Other fitters fit every row; DNest4 uses from min_rows of
    them (0 by default) to all, with a uniform prior on their number,
    adding and removing rows (new ones drawn from the priors), and gives
    the posterior of their number. The rows it does not use get the empty
    row of their table (e.g. a blob of amplitude 0).

    Only the rows of a table that share all their settings are
    interchangeable, and only they can be dropped: a row with settings of
    its own (see ParamSpace.own_settings: e.g. 'pt1_r[0]') is always
    used, and so are the rows whose empty columns are not free (e.g.
    blobs of a fixed amplitude), and all those of a table of a parameter
    with a mode.

    The free elements take a prior (or min and max for a uniform prior;
    see FitParamPrior). DNest4 evaluates one point at a time, and saves a
    particle every save_interval steps. It gives the posterior (the saved
    particles, weighted, with NaN in the rows a particle does not use),
    the posterior of the number of rows of each table (diagnostics
    'rows': the probability of each number of rows used, from none to
    all), and the evidence (without priors on several parameters; with
    them, it is not given). Its best solution is the point with the
    highest likelihood it evaluated. It reports its progress as the
    stage of the fit ('level 12, save 300 of 4000').

    Parameters
    ----------
    particles : int, optional
        The number of particles.
    new_level_interval : int, optional
        The number of points above the top level from which DNest4 makes
        the next level (more: more accurate levels, at more cost).
    save_interval : int, optional
        The steps between two saved particles.
    thread_steps : int, optional
        The steps between two updates of the levels.
    max_levels : int, optional
        The number of levels; 0 (by default): as many as the evidence
        needs (more with negative numbers: see DNest4).
    backtracking : float, optional
        How far the particles go back down the levels (lambda in Brewer
        et al. 2011).
    beta : float, optional
        How strongly the visits of the levels are made even, once they
        are all made (beta in Brewer et al. 2011).
    max_saves : int, optional
        The number of saved particles at which the run ends.
    min_rows : Mapping[str, int], optional
        The least rows used of tables, by the name of the table: that of
        its parameters without their column (e.g. 'pt1' of 'pt1_r'); 0
        by default.
    verbose : bool, optional
        Whether to print the progress: the levels, the saves, the
        evidence so far, and the posterior of the number of rows of each
        table.
    seed : int, optional
        The seed of the random numbers.
    uncertainty : UncertaintyOptions, optional
        How to work out the uncertainty of the best solution.

    Raises
    ------
    ConfigError
        If a number of particles, steps, saves or new_level_interval is
        not positive, backtracking is not positive, beta is negative, or
        a value of min_rows is negative.
    """

    @staticmethod
    def type() -> str:
        return 'dnest4'

    @staticmethod
    def fit_param_class() -> FitParamClass:
        return FitParamPrior

    def libraries(self) -> tuple[str, ...]:
        # (DNest4 is built into gbkfit)
        return ()

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), particles=self._particles,
            new_level_interval=self._new_level_interval,
            save_interval=self._save_interval,
            thread_steps=self._thread_steps, max_levels=self._max_levels,
            backtracking=self._backtracking, beta=self._beta,
            max_saves=self._max_saves, min_rows=self._min_rows,
            verbose=self._verbose) | self.dump_common()

    def __init__(
            self,
            particles: int = 5,
            new_level_interval: int = 5000,
            save_interval: int = 500,
            thread_steps: int = 100,
            max_levels: int = 0,
            backtracking: float = 10.0,
            beta: float = 100.0,
            max_saves: int = 4000,
            min_rows: Mapping[str, int] | None = None,
            verbose: bool = False,
            seed: int = 0,
            uncertainty: UncertaintyOptions | None = None
    ):
        super().__init__(seed, uncertainty)
        for name, value in dict(
                particles=particles, new_level_interval=new_level_interval,
                save_interval=save_interval, thread_steps=thread_steps,
                backtracking=backtracking, max_saves=max_saves).items():
            if not value > 0:
                raise ConfigError(
                    f"option '{name}' must be positive; it is {value}")
        if not beta >= 0:
            raise ConfigError(
                f"option 'beta' must be at least 0; it is {beta}")
        min_rows = dict(min_rows or {})
        if negative := [k for k, v in min_rows.items() if not v >= 0]:
            raise ConfigError(
                f"option 'min_rows' must be at least 0; it is not for "
                f"{negative}")
        self._particles = particles
        self._new_level_interval = new_level_interval
        self._save_interval = save_interval
        self._thread_steps = thread_steps
        self._max_levels = max_levels
        self._backtracking = backtracking
        self._beta = beta
        self._max_saves = max_saves
        self._min_rows = min_rows
        self._verbose = verbose

    def _run(
            self,
            problem: FitProblem,
            fit_params: list[FitParam],
            rng: np.random.Generator
    ) -> FitRun:
        layout = table_layout(
            problem.params(), problem.names(),
            [float(p.prior().ppf(0.5)) for p in fit_params], self._min_rows)
        unused_values = layout.unused_values()
        stopped = False
        with tempfile.TemporaryDirectory(prefix='gbkfit-dnest4-') as path:
            files = DNest4Files(path)
            progress = _Progress(
                problem, files, layout, self._max_saves, self._verbose)

            # (of the coordinates of the fixed elements, and of the rows
            # used of each table)
            def log_likelihood(
                    fixed: np.ndarray,
                    rows: list[np.ndarray]
            ) -> float:
                progress()
                x = problem.prior_transform(layout.place(fixed, rows, 0.5))
                unused = layout.unused([len(r) for r in rows])
                x[unused] = unused_values[unused]
                value = problem.nested_log_likelihood(x)
                return value if math.isfinite(value) else _IMPOSSIBLE

            try:
                _dnest4.run(
                    len(layout.fixed), [len(t.columns) for t in layout.tables],
                    [t.rows() for t in layout.tables], log_likelihood,
                    particles=self._particles,
                    new_level_interval=self._new_level_interval,
                    save_interval=self._save_interval,
                    thread_steps=self._thread_steps,
                    max_levels=self._max_levels, lambda_=self._backtracking,
                    beta=self._beta, max_saves=self._max_saves,
                    seed=int(rng.integers(2 ** 31)),
                    sample_file=files.sample(),
                    sample_info_file=files.sample_info(),
                    levels_file=files.levels())
            except FitStopped:
                stopped = True
            output = files.read()
        nsaves = len(output.info)
        if not nsaves:
            raise FitStopped("dnest4 was stopped before its first save")
        post = posterior(output, rng)
        samples, unused, counts = _samples(problem, layout, output)
        best = problem.best()
        if best is None:
            raise RuntimeError("dnest4 found no point with a finite chi2")
        evidence = None
        if not problem.has_joint_priors():
            evidence = Evidence(post.log_z, post.log_z_error)
        message = f'stopped after {nsaves} of {self._max_saves} saves' \
            if stopped else f'ran {nsaves} saves'
        return FitRun(
            solutions=[best[0]],
            success=not stopped,
            message=message,
            samples=samples,
            unused=unused,
            weights=post.weights,
            evidence=evidence,
            diagnostics=dict(
                rows=_rows(layout, counts, post.weights),
                levels=len(output.levels), saves=nsaves,
                information=post.information, ess=post.ess()),
            raw=output)


class _Progress:
    """
    The progress of a run of DNest4, from its files, at most every
    _PROGRESS_INTERVAL seconds: as the stage of the problem, and printed
    in full if verbose (see FitterDNest4).
    """

    def __init__(
            self,
            problem: FitProblem,
            files: DNest4Files,
            layout: TableLayout,
            max_saves: int,
            verbose: bool
    ):
        self._problem = problem
        self._files = files
        self._layout = layout
        self._max_saves = max_saves
        self._verbose = verbose
        self._last = time.perf_counter()

    def __call__(self) -> None:
        now = time.perf_counter()
        if now - self._last < _PROGRESS_INTERVAL:
            return
        self._last = now
        output = self._files.read()
        if not len(output.info):
            return
        self._problem.set_stage(
            f'level {len(output.levels) - 1}, save {len(output.info)} of '
            f'{self._max_saves}')
        if not self._verbose:
            return
        # (the weights at even steps of log X: no random numbers of the
        # fit are used)
        post = posterior(output, None)
        _, counts = _coordinates(self._layout, output.samples)
        rows = ', '.join(
            f'{name} {np.round(p, 3).tolist()}' for name, p in
            _rows(self._layout, counts, post.weights).items())
        print(
            f'dnest4: {self._problem.nevals()} evaluations, '
            f'{len(output.levels) - 1} levels (top log-likelihood '
            f'{output.levels[-1, 1]:.2f}), {len(output.info)} saves, '
            f'log Z {post.log_z:.2f}, ESS {post.ess():.0f}'
            + (f'; P(rows): {rows}' if rows else ''), flush=True)


def _coordinates(
        layout: TableLayout,
        samples: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the coordinates of the free elements of the saved particles
    (one per row, 0.5 in the rows of a table a particle does not use),
    and the number of rows of each table each uses (particles, tables).
    """
    nfixed = len(layout.fixed)
    coordinates = np.empty((len(samples), layout.ndim))
    counts = np.empty((len(samples), len(layout.tables)), dtype=int)
    for i, sample in enumerate(samples):
        start = nfixed
        rows = []
        for table in layout.tables:
            size = table.rows() * len(table.columns)
            cells = sample[start + 1:start + 1 + size]
            rows.append(cells.reshape(table.rows(), -1)[:int(sample[start])])
            start += 1 + size
        coordinates[i] = layout.place(sample[:nfixed], rows, 0.5)
        counts[i] = [len(r) for r in rows]
    return coordinates, counts


def _samples(
        problem: FitProblem,
        layout: TableLayout,
        output: DNest4Output
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the free values of the saved particles (one per row, those of
    an unused row of a table in the rows a particle does not use, as its
    likelihood was evaluated), which free elements each does not use, and
    the number of rows of each table each uses (particles, tables).
    """
    coordinates, counts = _coordinates(layout, output.samples)
    samples = problem.prior_transform(coordinates)
    unused = np.array([layout.unused(count) for count in counts])
    samples[unused] = np.broadcast_to(
        layout.unused_values(), samples.shape)[unused]
    return samples, unused, counts


def _rows(
        layout: TableLayout,
        counts: np.ndarray,
        weights: np.ndarray
) -> dict[str, list[float]]:
    """
    Return the posterior of the number of rows used of each table: the
    probability of each number, from none to all its rows.
    """
    return {
        table.name: np.bincount(
            table.always + counts[:, i], weights=weights,
            minlength=table.always + table.rows() + 1).tolist()
        for i, table in enumerate(layout.tables)}
