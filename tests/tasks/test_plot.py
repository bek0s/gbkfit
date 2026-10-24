"""
Tests for the plot task (gbkfit-cli plot), on a run made here: the
figures of each data item, and the corner plots.
"""

import numpy as np
import pytest
from astropy.table import Table

from gbkfit import outputs
from gbkfit.tasks.plot import plot


def write_run(directory, task, with_posterior=False):
    """
    Write a run with the outputs of a fit of a map, a cube, region values
    and a profile (without its result and samples, if the task is eval).
    """
    rng = np.random.default_rng(1)
    y, x = np.indices((16, 20))
    disk = np.exp(-((x - 10) ** 2 + (y - 8) ** 2) / 20)
    items = dict(
        moment1=disk * 100,
        spectra=np.stack([disk * (z + 1) for z in range(6)]),
        values=np.linspace(1, 2, 7))
    run = outputs.create(str(directory), task, 'overwrite')
    for key, model in items.items():
        data = model + rng.normal(0, 0.1, model.shape)
        for kind, value in (
                ('data', data), ('error', np.full_like(data, 0.1)),
                ('model', model), ('residual', (model - data) / 0.1)):
            run.write(
                f'observations/observation0/{kind}/{key}.fits', kind, value,
                observation='observation0', item=key)
    # (and a profile, as a table of the values at radii)
    radii = np.linspace(1, 10, 8)
    for kind, value in (
            ('data', 150 * radii / (radii + 2)), ('error', np.full(8, 5.0)),
            ('model', 148 * radii / (radii + 2)), ('residual', np.zeros(8))):
        run.write(
            f'observations/rc/{kind}/vcirc.ecsv', kind,
            Table(dict(radius=radii, value=value)), observation='rc',
            item='vcirc')
    psf = np.exp(-((x - 10) ** 2 / 2 + (y - 8) ** 2 / 8))
    run.write(
        'observations/observation0/extra/psf_lo.fits', 'extra', psf,
        observation='observation0', name='psf_lo')
    if task == 'fit':
        elements = dict(a=1.0, b=2.0, c=3.0)
        solution = dict(free=dict(a=1.0, b=2.0), elements=elements)
        run.write(
            'fit/result.json', 'fit_result',
            dict(solutions=[solution, solution]))
        samples = Table(
            list(rng.normal([1, 2, 3], 0.1, (500, 3)).T),
            names=['a', 'b', 'c'])
        if with_posterior:
            samples['weight'] = np.full(500, 1 / 500)
            run.write('fit/posterior.ecsv', 'posterior', samples)
        for k in range(2):
            run.write(
                f'fit/solutions/{k}/laplace.ecsv', 'laplace', samples,
                solution=str(k))
    run.close()


def figures(directory):
    """Return the paths of the figures of the run of the plot task."""
    return sorted(p.path for p in outputs.open(directory).products('figure'))


ITEM_FIGURES = [
    f'observations/observation0/fit/{key}.png'
    for key in ('moment1', 'spectra', 'values')] + [
    'observations/rc/fit/vcirc.png']


@pytest.mark.parametrize('with_posterior', [True, False])
def test_figures_of_each_item_and_of_the_posterior(tmp_path, with_posterior):
    write_run(tmp_path, 'fit', with_posterior)
    plot(str(tmp_path), None, 'overwrite', 'png', 50, False, 'corner', None)
    posterior = ['fit/posterior.png'] if with_posterior \
        else ['fit/solutions/0/laplace.png', 'fit/solutions/1/laplace.png']
    assert figures(tmp_path / 'plots') == sorted(ITEM_FIGURES + posterior)
    run = outputs.open(tmp_path / 'plots')
    assert run.options['source'] == '..'
    figure = run.product('figure', plot='fit', item='spectra')
    assert 'observations/observation0/model/spectra.fits' in figure.sources


def test_separate_corner_plots_of_chosen_elements(tmp_path):
    write_run(tmp_path, 'fit', True)
    plot(str(tmp_path), str(tmp_path / 'figures'), 'overwrite', 'png', 50,
         True, 'separate', ['a', 'c', 'b'])
    run = outputs.open(tmp_path / 'figures')
    corners = sorted(p.path for p in run.products('figure', plot='posterior'))
    assert corners == [
        'fit/posterior/a_b.png', 'fit/posterior/a_c.png',
        'fit/posterior/c_b.png']
    with pytest.raises(ValueError, match="not in the samples"):
        plot(str(tmp_path), str(tmp_path / 'figures'), 'overwrite', 'png',
             50, True, 'corner', ['d'])


def test_an_evaluation_of_the_objective_has_no_posterior(tmp_path):
    write_run(tmp_path, 'eval')
    plot(str(tmp_path), None, 'overwrite', 'png', 50, False, 'corner', None)
    assert figures(tmp_path / 'plots') == ITEM_FIGURES
