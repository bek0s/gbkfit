"""
Tests of the DNest4 fitter: which rows of a table of blobs it varies,
how its coordinates make the free values, and its posterior and evidence
on likelihoods of known integrals.
"""

import math
import threading

import numpy as np
import pytest

from gbkfit.fitting import FitControl
from gbkfit.fitting.fitters._tables import table_layout
from gbkfit.fitting.fitters.dnest4 import FitterDNest4, _dnest4
from gbkfit.fitting.fitters.dnest4._output import DNest4Files, posterior
from gbkfit.model import galaxy_parser
from gbkfit.params import ParamSpace
from gbkfit.utils.parseutils import ConfigError


GALAXY = dict(components=[dict(
    type='disk', loose=False, tilted=False, rnodes=[0, 5, 10],
    density=dict(ptraits=[
        dict(type='exponential'), dict(type='mixture_gauss', nblobs=3)]),
    emission=dict(type='band'))])

PROPERTIES = dict(
    xpos=0, ypos=0, posa=30, incl=60, pt_a=dict(min=0, max=2), pt_s=2,
    pt1_r=dict(min=0, max=10), pt1_t=dict(min=0, max=360),
    pt1_a=dict(min=0, max=2), pt1_s=1, pt1_q=1, pt1_p=0)


def layout(properties, min_rows=None):
    """The layout of the galaxy with the given properties."""
    params = ParamSpace(galaxy_parser.load(GALAXY).pdescs(), properties)
    names = tuple(params.free_properties())
    medians = [1.0] * len(names)
    return table_layout(params, names, medians, min_rows or {}), names


def test_rows_that_share_their_settings_are_varied():
    lay, names = layout(PROPERTIES)
    (table,) = lay.tables
    assert table.name == 'pt1' and table.columns == ('r', 't', 'a')
    assert [[names[i] for i in row] for row in table.elements] == [
        [f'pt1_{c}[{i}]' for c in 'rta'] for i in range(3)]
    assert [names[i] for i in lay.fixed] == ['pt_a']
    # Rows with settings of their own are always used
    own = {k: v for k, v in PROPERTIES.items() if k != 'pt1_r'} | {
        'pt1_r[0]': dict(min=2, max=3), 'pt1_r[1:]': dict(min=0, max=10)}
    lay, names = layout(own)
    assert not lay.tables
    assert 'pt1_r[0]' in [names[i] for i in lay.fixed]
    # Blobs of a fixed amplitude cannot be dropped
    assert not layout(PROPERTIES | dict(pt1_a=1))[0].tables


def test_min_rows_keeps_the_first_rows_always_used():
    lay, names = layout(PROPERTIES, dict(pt1=1))
    (table,) = lay.tables
    assert table.always == 1 and table.rows() == 2
    assert [names[i] for i in lay.fixed] == [
        'pt_a', 'pt1_r[0]', 'pt1_t[0]', 'pt1_a[0]']
    assert not layout(PROPERTIES, dict(pt1=3))[0].tables
    with pytest.raises(ConfigError, match='which has 3'):
        layout(PROPERTIES, dict(pt1=4))
    with pytest.raises(ConfigError, match='cannot vary'):
        layout(PROPERTIES, dict(pt2=1))


def test_rows_used_go_first_and_the_others_get_the_empty_row():
    lay, names = layout(PROPERTIES)
    values = lay.place(np.array([1.5]), [np.array([[1.0, 10.0, 0.5]])], 0.5)
    unused = lay.unused([1])
    values[unused] = lay.unused_values()[unused]
    values = dict(zip(names, values))
    assert values['pt_a'] == 1.5
    assert [values[f'pt1_a[{i}]'] for i in range(3)] == [0.5, 0.0, 0.0]
    assert [values[f'pt1_r[{i}]'] for i in range(3)] == [1.0, 1.0, 1.0]


def test_posterior_of_the_number_of_rows_and_the_evidence(tmp_path):
    # Two coordinates of a normalised Gaussian likelihood (integral 1 in
    # the unit cube) and a table of up to 3 rows whose likelihood is
    # 2^rows: P(rows) is 2^rows / 15, and log Z log(15 / 4)
    sigma = 0.1

    def log_likelihood(fixed, rows):
        (table,) = rows
        gauss = -0.5 * ((fixed - 0.5) / sigma) ** 2 \
            - math.log(math.sqrt(2 * math.pi) * sigma)
        return float(gauss.sum() + len(table) * math.log(2))

    files = DNest4Files(str(tmp_path))
    _dnest4.run(
        2, [1], [3], log_likelihood, particles=5, new_level_interval=2000,
        save_interval=200, thread_steps=100, max_levels=0, lambda_=10,
        beta=100, max_saves=3000, seed=1, sample_file=files.sample(),
        sample_info_file=files.sample_info(), levels_file=files.levels())
    output = files.read()
    post = posterior(output, np.random.default_rng(0))
    assert post.log_z == pytest.approx(math.log(15 / 4), abs=0.15)
    counts = output.samples[:, 2].astype(int)
    p = np.bincount(counts, weights=post.weights, minlength=4)
    assert p == pytest.approx(np.array([1, 2, 4, 8]) / 15, abs=0.04)


def test_samplers_run_in_several_threads_at_once(tmp_path):
    # (each waits for the other at its first evaluation: both run at once)
    started = set()
    both_started = threading.Event()

    def log_likelihood(fixed, rows):
        if threading.get_ident() not in started:
            started.add(threading.get_ident())
            if len(started) == 2:
                both_started.set()
            both_started.wait(timeout=10)
        return -float(fixed[0])

    def run(name, saves):
        files = DNest4Files(str(tmp_path / name))
        (tmp_path / name).mkdir()
        _dnest4.run(
            1, [], [], log_likelihood, particles=2,
            new_level_interval=100, save_interval=10, thread_steps=10,
            max_levels=5, lambda_=10, beta=100, max_saves=saves, seed=1,
            sample_file=files.sample(), sample_info_file=files.sample_info(),
            levels_file=files.levels())
        results[name] = len(files.read().info)

    results = {}
    threads = [threading.Thread(target=run, args=(f'run{i}', 50 * (i + 1)))
               for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == dict(run0=50, run1=100)


class StopAfter(FitControl):
    """A control that stops a fit after n evaluations."""

    def __init__(self, n):
        super().__init__()
        self._n = n

    def stopped(self):
        self._n -= 1
        return self._n < 0


def blob_objective():
    """An objective of a mock image of the galaxy, with one blob."""
    from gbkfit.dataset import Data, DatasetPixelBrightness
    from gbkfit.driver.drivers.host import DriverHost
    from gbkfit.instrument import Instrument, PSFGauss
    from gbkfit.objective import Objective
    from gbkfit.observation import (
        Flat, ObservationGroup, ObservationSky, PixelBrightness)

    def group(dataset):
        return ObservationGroup(galaxy_parser.load(GALAXY), ObservationSky(
            PixelBrightness.from_data(dataset), Flat(),
            instrument=Instrument(psf=PSFGauss(1.5)), data=dataset)
        ).plan(DriverHost())
    shape = (16, 16)
    image = group(DatasetPixelBrightness(Data(np.zeros(shape))))
    truth = {k: v for k, v in PROPERTIES.items() if not isinstance(v, dict)}
    truth |= dict(pt_a=1.0, pt1_r=[3, 5, 5], pt1_t=[30, 0, 0],
                  pt1_a=[1, 0, 0])
    model = image.model_h(
        ParamSpace(image.group().pdescs(), truth).evaluate())[0]
    model = model['brightness']['d']
    sigma = 0.02 * model.max()
    noise = np.random.default_rng(1).normal(0, sigma, shape)
    return Objective(group(DatasetPixelBrightness(
        Data(model + noise, error=np.full(shape, sigma)))))


def test_a_stopped_fit_keeps_its_saves():
    objective = blob_objective()
    params = ParamSpace(objective.pdescs(), PROPERTIES)
    fitter = FitterDNest4(
        new_level_interval=200, save_interval=50, max_saves=1000)
    result = fitter.fit(objective, params, control=StopAfter(2000))
    assert not result.success and 'stopped after' in result.message
    rows = result.diagnostics['rows']['pt1']
    assert len(rows) == 4 and sum(rows) == pytest.approx(1)
    # The rows a sample does not use are NaN
    samples = result.posterior.samples
    a = samples[:, [result.posterior.names.index(f'pt1_a[{i}]')
                    for i in range(3)]]
    used = ~np.isnan(a)
    assert np.all(used[:, :-1] >= used[:, 1:])
    # It has every particle, whatever the number of rows it uses: the
    # weights of those of k rows are the probability of k rows
    weights = result.posterior.weights
    nrows = np.sum(used, axis=1)
    assert [np.sum(weights[nrows == k]) / np.sum(weights)
            for k in range(4)] == pytest.approx(rows)
    assert math.isfinite(result.evidence.log_z)
