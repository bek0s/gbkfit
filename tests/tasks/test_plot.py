"""
Tests for the plot task (gbkfit-cli plot), on an output directory made
here: the figures of each data item, and the corner plots.
"""

import json

import astropy.io.fits as fits
import numpy as np
import pytest

from gbkfit.tasks.plot import plot


def write_result(directory, with_posterior):
    """Write the outputs of a fit of a map, a cube and region values."""
    rng = np.random.default_rng(1)
    y, x = np.indices((16, 20))
    disk = np.exp(-((x - 10) ** 2 + (y - 8) ** 2) / 20)
    items = dict(
        moment1=(disk * 100,),
        spectra=(np.stack([disk * (z + 1) for z in range(6)]),),
        values=(np.linspace(1, 2, 7),))
    for key, (model,) in items.items():
        data = model + rng.normal(0, 0.1, model.shape)
        fits.writeto(directory / f'data_0_{key}_d.fits', data)
        fits.writeto(
            directory / f'data_0_{key}_e.fits', np.full_like(data, 0.1))
        fits.writeto(directory / f'model_0_{key}_d.fits', model)
        fits.writeto(
            directory / f'residual_{key}_d.fits', (model - data) / 0.1)
    psf = np.exp(-((x - 10) ** 2 / 2 + (y - 8) ** 2 / 8))
    fits.writeto(directory / 'model_extra_observation0_psf_lo.fits', psf)
    elements = dict(a=1.0, b=2.0, c=3.0)
    solution = dict(free=dict(a=1.0, b=2.0), elements=elements)
    (directory / 'result.json').write_text(json.dumps(dict(
        solutions=[solution, solution])))
    samples = rng.normal([1, 2, 3], 0.1, (500, 3))
    names = 'a,b,c'
    if with_posterior:
        rows = np.column_stack((samples, np.full(500, 1 / 500)))
        np.savetxt(directory / 'posterior.csv', rows, delimiter=',',
                   header=names + ',weight', comments='')
    for k in range(2):
        np.savetxt(directory / f'solution_{k}_laplace.csv', samples,
                   delimiter=',', header=names, comments='')


@pytest.mark.parametrize('with_posterior', [True, False])
def test_figures_of_each_item_and_of_the_posterior(tmp_path, with_posterior):
    write_result(tmp_path, with_posterior)
    plot(str(tmp_path), str(tmp_path / 'figures'), 'overwrite', 'png', 50,
         False, 'corner', None)
    names = sorted(p.name for p in (tmp_path / 'figures').iterdir())
    posterior = ['posterior.png'] if with_posterior \
        else ['laplace_0.png', 'laplace_1.png']
    assert names == sorted([
        'fit_0_moment1.png', 'fit_0_spectra.png', 'fit_0_values.png',
        *posterior])


def test_separate_corner_plots_of_chosen_elements(tmp_path):
    write_result(tmp_path, True)
    plot(str(tmp_path), str(tmp_path / 'figures'), 'overwrite', 'png', 50,
         True, 'separate', ['a', 'c', 'b'])
    corners = sorted(
        p.name for p in (tmp_path / 'figures').glob('posterior_*'))
    assert corners == [
        'posterior_a_b.png', 'posterior_a_c.png', 'posterior_c_b.png']
    with pytest.raises(ValueError, match="not in posterior.csv"):
        plot(str(tmp_path), str(tmp_path / 'figures'), 'overwrite', 'png',
             50, True, 'corner', ['d'])
