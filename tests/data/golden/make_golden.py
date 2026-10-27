"""
Make the golden cases of this directory, and their golden values.

A case is a folder: config.yaml (galaxies and their observations, with
the parameters at trial values), the data of its observations, and
golden_<driver>.json for each driver (host, cuda): what the code computed
for the configuration at the trial values (see golden_values). The data
are mock data of the model at the true values, with noise of a fixed
seed: made once from the recipe of the case (see RECIPE_MODULES)
and frozen.

tests/model/test_golden.py computes the golden values again and compares
them with those stored (see compare): a change that must not change the
results (e.g. a refactor) is checked not to, end to end, from a
configuration to its chi-square.

Usage:

    python make_golden.py data [--root DIR] [CASE ...]
    python make_golden.py config [--root DIR] [CASE ...]
    python make_golden.py values [--root DIR] [--driver NAME] [CASE ...]

'data' writes the configuration and the data of cases from their recipes
(all cases by default): for new cases, or recipes that change; their
golden values must then be made again. 'config' writes the configuration
of cases from their recipes and keeps their data: after a change of the
format of configurations, which must not change their golden values.
'values' writes the golden values
of cases with the current code (both drivers by default): after a change
that changes the results on purpose, which its commit says. --root
writes or reads the cases in another directory (e.g. to try a recipe).
"""

import argparse
import dataclasses
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

if TYPE_CHECKING:
    from gbkfit.setup import Setup


HERE = pathlib.Path(__file__).parent

DRIVERS = ('host', 'cuda')

# The relative tolerance of the values of float32 and float64
# observations, by driver (the CUDA kernels that add with atomics add in
# an order that changes from run to run)
TOLERANCES = {
    ('float32', 'host'): 1e-5, ('float32', 'cuda'): 1e-4,
    ('float64', 'host'): 1e-9, ('float64', 'cuda'): 1e-9}

# The seed of the weights of the checksums of arrays (see summary)
_CHECKSUM_SEED = 20261024


@dataclasses.dataclass(frozen=True)
class Case:
    """
    The recipe of a golden case.

    Attributes
    ----------
    name : str
        The name of its folder.
    doc : str
        What it covers.
    galaxies : list of dict
        The section 'galaxies' of its configuration.
    observations : list of dict
        The section 'observations', without data: those of the observables
        give their grids.
    truth : dict
        The parameter properties of its mock data (all fixed), of its
        galaxies (or those of mock_galaxies).
    trial : dict
        The parameter properties of its configuration: the values at
        which it is evaluated (free elements at their 'value').
    noises : list of dict
        The noise of the mock data of each observation (see
        gbkfit.observation.noise_parser).
    likelihoods : list of dict or None, optional
        The likelihood of each observation (none: the default).
    data_options : list of dict, optional
        Options added to the data of each observation in its
        configuration (e.g. the correlation of its noise).
    params : dict, optional
        Other options of the section 'params' (e.g. priors, derived).
    sections : dict, optional
        Other sections of its configuration (e.g. pdescs, depth_order).
    prepare : callable, optional
        Writes the files the recipe needs into the folder of the case
        (e.g. an image of an image trait), before the data are made.
    seed : int, optional
        The seed of the noise.
    nsamples : int, optional
        The number of draws of the errors of measured data (see
        ObservationGroup.mock).
    mock_galaxies : list of dict, optional
        The galaxies of its mock data, if not those of its configuration
        (e.g. Jeans models of the stars of Schwarzschild models, whose
        observations have no mock data).
    """
    name: str
    doc: str
    galaxies: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    truth: dict[str, Any]
    trial: dict[str, Any]
    noises: list[dict[str, Any]]
    likelihoods: list[dict[str, Any] | None] | None = None
    data_options: list[dict[str, Any]] | None = None
    params: dict[str, Any] = dataclasses.field(default_factory=dict)
    sections: dict[str, Any] = dataclasses.field(default_factory=dict)
    prepare: Callable[[pathlib.Path], None] | None = None
    seed: int = 1
    nsamples: int = 30
    mock_galaxies: list[dict[str, Any]] | None = None


# --- The recipes ----------------------------------------------------------

# The modules of the recipes of the cases, by kind: each has CASES, the
# options of each Case
RECIPE_MODULES = (
    'recipes_components', 'recipes_observations', 'recipes_mass')


def recipes() -> dict[str, Case]:
    """Return the recipes of all cases, by name."""
    cases = {}
    for name in RECIPE_MODULES:
        spec = importlib.util.spec_from_file_location(
            name, HERE / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for options in module.CASES:
            case = Case(**options)
            if case.name in cases:
                raise ValueError(f"two cases are named {case.name!r}")
            cases[case.name] = case
    return cases


# --- Making the data -------------------------------------------------------

def _load(config: dict[str, Any], root: pathlib.Path, driver: str) -> 'Setup':
    """
    Set up a configuration whose relative paths are relative to root, on a
    driver (see gbkfit.load).
    """
    import gbkfit
    from gbkfit.project import FORMAT_VERSION
    return gbkfit.load(dict(
        config, version=FORMAT_VERSION, root=str(root),
        driver=dict(type=driver)))


def _rebased(info: Any, prefix: str) -> Any:
    """Return options with the strings that start with prefix without it."""
    if isinstance(info, str):
        return info.removeprefix(prefix)
    if isinstance(info, list):
        return [_rebased(item, prefix) for item in info]
    if isinstance(info, dict):
        return {k: _rebased(v, prefix) for k, v in info.items()}
    return info


def _compacted(info: Any, folder: pathlib.Path) -> Any:
    """
    Return the options of data with their files made small: the data in
    float32, an error map of one value as that value, and a mask that
    masks nothing left out (their files removed).
    """
    from astropy.io import fits
    if isinstance(info, list):
        return [_compacted(item, folder) for item in info]
    if not isinstance(info, dict):
        return info
    info = {k: _compacted(v, folder) for k, v in info.items()}
    if not isinstance(info.get('data'), str):
        return info
    for key in ('data', 'error', 'mask'):
        if not isinstance(info.get(key), str):
            continue
        path = folder / info[key]
        with fits.open(path) as hdus:
            array = np.asarray(hdus[0].data)
            header = hdus[0].header.copy()
        finite = array[np.isfinite(array)]
        if key == 'error' and finite.size == array.size \
                and np.all(finite == finite[0]):
            info[key] = float(finite[0])
            path.unlink()
        elif key == 'mask' and np.all(array == 1):
            del info[key]
            path.unlink()
        else:
            fits.writeto(
                path, array.astype(np.float32), header, overwrite=True)
    return info


def make_data(case: Case, folder: pathlib.Path) -> None:
    """
    Write the configuration and the data of a case into its folder: mock
    data of the model at its true values (see Case.mock_galaxies), and
    its parameters at its trial values.
    """
    from gbkfit.observation import likelihood_parser, noise_parser
    from gbkfit.observation import observation_parser
    from gbkfit.project import FORMAT_VERSION
    from gbkfit.utils import numutils, parseutils
    folder.mkdir(parents=True, exist_ok=True)
    if case.prepare is not None:
        case.prepare(folder)
    sections = dict(galaxies=case.galaxies) | case.sections
    setup = _load(
        sections | dict(
            galaxies=case.mock_galaxies or case.galaxies,
            observations=case.observations,
            params=dict(properties=case.truth)), folder, 'host')
    group_plan, params = setup.group_plan, setup.params
    likelihoods = case.likelihoods or [None] * len(case.observations)
    # (their files, e.g. error maps or the modules of user likelihoods,
    # are in the folder of the case)
    with parseutils.paths_relative_to(str(folder)):
        noises = [noise_parser.load(noise) for noise in case.noises]
        likelihoods = [
            None if info is None else likelihood_parser.load(info)
            for info in likelihoods]
    datasets = group_plan.mock(
        params.evaluate(), noises, nsamples=case.nsamples, seed=case.seed)
    data_options = case.data_options or [{}] * len(case.observations)
    prefix = str(folder) + os.sep
    group = group_plan.group()
    observations = []
    for i, observation in enumerate(group.observations()):
        mocked = observation.with_data(datasets[i][0], likelihoods[i])
        info = _rebased(observation_parser.dump(
            mocked, f'{prefix}{group.observation_label(i)}_',
            overwrite=True), prefix)
        info['data'] = _compacted(info['data'], folder) | data_options[i]
        observations.append(info)
    config = dict(version=FORMAT_VERSION, driver=dict(type='host')) \
        | sections | dict(
        observations=observations,
        params=dict(properties=case.trial) | case.params)
    _write_config(case, folder, numutils.nativify(config))


def make_config(case: Case, folder: pathlib.Path) -> None:
    """
    Write the configuration of a case into its folder from its recipe,
    with the data of the case as they are: that of new data (made in a
    temporary folder), with the options of the data of each observation
    of the case (migrated to the current format).
    """
    from gbkfit import migration
    with tempfile.TemporaryDirectory() as temporary:
        make_data(case, pathlib.Path(temporary))
        with open(pathlib.Path(temporary, 'config.yaml')) as f:
            config = yaml.safe_load(f)
    with open(folder / 'config.yaml') as f:
        current = yaml.safe_load(f)
    migration.migrate(current)
    for observation, data in zip(
            config['observations'], current['observations'], strict=True):
        observation['data'] = data['data']
    _write_config(case, folder, config)


def _write_config(
        case: Case, folder: pathlib.Path, config: dict[str, Any]
) -> None:
    """Write the configuration of a case into its folder."""
    with open(folder / 'config.yaml', 'w') as f:
        f.write(f"# {case.name}: {case.doc}\n")
        f.write("# (written by make_golden.py; do not edit)\n")
        yaml.safe_dump(
            config, f, sort_keys=False, default_flow_style=None, width=78)


# --- The golden values ---------------------------------------------------

def summary(array: Any) -> dict[str, Any]:
    """
    Return a summary of an array that changes when any of its values
    does: its size, its number of finite values, the sum, the sum of the
    absolute values and the sum of the squares of its finite values, and
    the checksum: the sum of its finite values times fixed random weights
    in [0.5, 1.5], and the scale of that sum (that of the absolute
    values), which sets its tolerance (see compare).
    """
    values = np.asarray(array, dtype=np.float64).ravel()
    finite = np.isfinite(values)
    weights = np.random.default_rng(_CHECKSUM_SEED).uniform(
        0.5, 1.5, values.size)
    values = np.where(finite, values, 0)
    return dict(
        size=int(values.size), finite=int(np.sum(finite)),
        sum=float(np.sum(values)), abs=float(np.sum(np.abs(values))),
        sumsq=float(np.sum(values ** 2)),
        checksum=float(np.sum(weights * values)),
        scale=float(np.sum(weights * np.abs(values))))


def golden_values(folder: pathlib.Path, driver: str) -> dict[str, Any]:
    """
    Return the golden values of a case on a driver: for each observation,
    a summary of each part of each item of its model (data, mask,
    weights), with the values that the objective solves, if any (the
    model of a fit); of the objective, the chi-square, the
    log-likelihood, the noise chi-square, the counts, and summaries of the
    residuals and of the solved values, if any; the summaries of the
    derived values; and the chi-square of the priors on several
    parameters, if any.
    """
    from gbkfit.fitting.problem import FitProblem
    from gbkfit.setup import read_config
    file = read_config(str(folder / 'config.yaml'))
    setup = _load(file.config, pathlib.Path(file.root), driver)
    group_plan, params = setup.group_plan, setup.params
    values = params.evaluate()
    with setup:
        objective = setup.objective
        evaluation = objective.evaluate(values)
        chi2, log_likelihood, _ = evaluation.merits
        models = group_plan.model_h(params.evaluate(
            solved=evaluation.solved))
        golden: dict[str, Any] = dict(model=[
            {key: {part: summary(array) for part, array in item.items()
                   if array is not None}
             for key, item in model.items()}
            for model in models])
        golden['objective'] = dict(
            chi2=chi2, log_likelihood=log_likelihood,
            noise_chi2=evaluation.noise_chi2,
            nresiduals=objective.nresiduals(),
            nindependent=objective.nindependent(),
            nnuisance=evaluation.nnuisance,
            # (of each observation: those of a user likelihood of the
            # log-likelihood are 0, and the objective has no residuals)
            residuals=[
                summary(r) for r in objective.residual_vector_h(values, True)],
            solved=summary(list(evaluation.solved.values()))
            if params.solves() else None)
        derived = params.derived_values(values, group_plan.functions())
        golden['derived'] = {
            name: summary(value) for name, value in derived.items()}
        if params.priors():
            free = np.array([
                p['value'] for p in params.free_properties().values()])
            golden['joint_priors_chi2'] = FitProblem(
                objective, params).prior_chi2(free)
    return golden


def tolerance(folder: pathlib.Path, driver: str) -> float:
    """
    Return the relative tolerance of the golden values of a case on a
    driver: that of the least precise dtype of its observations (those on
    the sky are float32 by default; profiles are float64).
    """
    with open(folder / 'config.yaml') as f:
        config = yaml.safe_load(f)
    dtypes = {
        'float64' if obs['type'] == 'profile' else obs.get('dtype', 'float32')
        for obs in config['observations']}
    return max(TOLERANCES[(dtype, driver)] for dtype in dtypes)


def _is_summary(value: Any) -> bool:
    return isinstance(value, Mapping) and 'checksum' in value


def _compare_summary(
        actual: Mapping[str, Any], golden: Mapping[str, Any],
        rtol: float, path: str
) -> list[str]:
    """Return the differences of two summaries (see summary) beyond rtol."""
    problems = [
        f"{path}.{key}: {actual[key]} != {golden[key]}"
        for key in ('size', 'finite') if actual[key] != golden[key]]
    # (sums of finite values at a relative tolerance of their scales; a
    # floor for arrays of zeros)
    floor = 1e-300
    for key, scale in (('sum', 'abs'), ('abs', 'abs'), ('sumsq', 'sumsq'),
                       ('checksum', 'scale'), ('scale', 'scale')):
        limit = 2 * rtol * abs(golden[scale]) + floor
        if not abs(actual[key] - golden[key]) <= limit:
            problems.append(
                f"{path}.{key}: {actual[key]!r} != {golden[key]!r} "
                f"(tolerance {limit:.3g})")
    return problems


def compare(
        actual: Any, golden: Any, rtol: float, path: str = 'golden'
) -> list[str]:
    """
    Return the differences between computed and stored golden values
    beyond the relative tolerance rtol: counts and sizes exactly, numbers
    relative to themselves, sums relative to their scales (see summary).
    """
    if _is_summary(golden):
        if not _is_summary(actual):
            return [f"{path}: {actual!r} is not a summary"]
        return _compare_summary(actual, golden, rtol, path)
    if isinstance(golden, Mapping):
        if not isinstance(actual, Mapping):
            return [f"{path}: {actual!r} is not a mapping"]
        if set(actual) != set(golden):
            return [f"{path}: keys {sorted(actual)} != {sorted(golden)}"]
        return [p for key in golden for p in compare(
            actual[key], golden[key], rtol, f"{path}.{key}")]
    if isinstance(golden, list):
        if not isinstance(actual, list) or len(actual) != len(golden):
            return [f"{path}: {actual!r} != {golden!r}"]
        return [p for i, (a, g) in enumerate(zip(actual, golden))
                for p in compare(a, g, rtol, f"{path}[{i}]")]
    if isinstance(golden, int) and not isinstance(golden, bool):
        return [] if actual == golden else [f"{path}: {actual} != {golden}"]
    if golden is None or actual is None:
        return [] if actual is golden \
            else [f"{path}: {actual!r} != {golden!r}"]
    limit = rtol * max(abs(golden), 1e-300)
    if not abs(actual - golden) <= limit:
        return [f"{path}: {actual!r} != {golden!r} (tolerance {limit:.3g})"]
    return []


def write_values(folder: pathlib.Path, driver: str) -> None:
    """Write the golden values of a case on a driver into its folder."""
    from gbkfit.utils import numutils
    values = numutils.nativify(golden_values(folder, driver))
    with open(folder / f'golden_{driver}.json', 'w') as f:
        json.dump(values, f, indent=1)
        f.write('\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=('data', 'config', 'values'))
    parser.add_argument('cases', nargs='*')
    parser.add_argument('--root', type=pathlib.Path, default=HERE)
    parser.add_argument('--driver', choices=DRIVERS, action='append')
    args = parser.parse_args()
    if args.what in ('data', 'config'):
        known = recipes()
        names = args.cases or list(known)
        if unknown := [name for name in names if name not in known]:
            sys.exit(f"unknown cases: {unknown}")
        make = make_data if args.what == 'data' else make_config
        for name in names:
            make(known[name], args.root / name)
            print(f"{name}: {args.what}")
        return
    names = args.cases or sorted(
        path.name for path in args.root.iterdir()
        if (path / 'config.yaml').exists())
    for name in names:
        for driver in args.driver or DRIVERS:
            write_values(args.root / name, driver)
            print(f"{name}: golden values on {driver}")


if __name__ == '__main__':
    main()
