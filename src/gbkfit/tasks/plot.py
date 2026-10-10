"""
The plot task: figures of the output directory of a fit (or of the eval
task in objective mode): the data, model and residuals of each data item,
and corner plots of the posterior.
"""

import itertools
import json
import logging
import os
import re
from collections.abc import Sequence
from typing import Literal

import astropy.io.fits as fits
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from . import _detail


__all__ = [
    'plot'
]


_log = logging.getLogger(__name__)

# The FITS outputs of the data: data_<observation>_<item>_d.fits
_DATA_FILE = re.compile(r'data_(\d+)_(.+)_d\.fits')

# The FITS output of a constant PSF: model_extra_<observation>_psf_lo.fits
_PSF_FILE = re.compile(r'model_extra_.+_psf_lo\.fits')

# The width of a Gaussian at half its maximum, in standard deviations
_FWHM = 2 * np.sqrt(2 * np.log(2))

# An ellipse: its width, height and angle (degrees), in pixels
_EllipseShape = tuple[float, float, float]


def plot(
        result_dir: str,
        output_dir: str,
        output_dir_mode: Literal['terminate', 'overwrite', 'unique'],
        format_: Literal['pdf', 'png'],
        dpi: int,
        only_best: bool,
        posterior_mode: Literal['none', 'corner', 'separate'],
        posterior_params: Sequence[str] | None
) -> None:
    """
    Make the figures of the output directory of a fit: for each data item
    of each observation, its data, model and residuals (fit_<i>_<item>),
    and corner plots of the posterior samples (posterior), or of the
    Laplace samples of the solutions (laplace_<k>).

    Maps are shown as images (with the PSF, if constant, in a corner),
    cubes as their moment 0 and the position-velocity slice through their
    central row, and values per region as lines.

    Parameters
    ----------
    result_dir : str
        The output directory of the fit.
    output_dir : str
        The directory of the figures.
    output_dir_mode : {'terminate', 'overwrite', 'unique'}
        What to do if the directory of the figures exists (see
        make_output_dir).
    format_ : {'pdf', 'png'}
        The format of the figures.
    dpi : int
        Their resolution.
    only_best : bool
        Whether to plot only the Laplace samples of the best solution
        (when there is no posterior).
    posterior_mode : {'none', 'corner', 'separate'}
        No posterior plots, one corner plot of the elements, or one of each
        pair of them.
    posterior_params : Sequence of str, optional
        The elements of the posterior plots (by default, the free ones).

    Raises
    ------
    ValueError
        If a posterior element is not in the samples.
    """
    output_dir = _detail.make_output_dir(output_dir, output_dir_mode)
    figures = _fit_figures(result_dir)
    if posterior_mode != 'none':
        figures |= _posterior_figures(
            result_dir, only_best, posterior_mode == 'separate',
            posterior_params)
    if not figures:
        _log.warning(f"found nothing to plot in '{result_dir}'")
    for name, figure in figures.items():
        filename = os.path.join(output_dir, f'{name}.{format_}')
        figure.savefig(filename, dpi=dpi, bbox_inches='tight')
    _log.info(f"wrote {len(figures)} figures to '{output_dir}'")


def _read(directory: str, filename: str) -> np.ndarray | None:
    """Return the data of a FITS file, or None if there is none."""
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        return None
    return np.asarray(fits.getdata(path), dtype=float)


def _fit_figures(result_dir: str) -> dict[str, Figure]:
    """Return the figures of the data, model and residuals of each item."""
    items = sorted(
        match.groups() for filename in os.listdir(result_dir)
        if (match := _DATA_FILE.fullmatch(filename)))
    psf = _psf_shape(result_dir) \
        if len({index for index, _ in items}) == 1 else None
    figures = {}
    for index, key in items:
        model = _read(result_dir, f'model_{index}_{key}_d.fits')
        if model is None:
            continue
        data = _read(result_dir, f'data_{index}_{key}_d.fits')
        error = _read(result_dir, f'data_{index}_{key}_e.fits')
        residual = _read(result_dir, f'residual_{index}_{key}_d.fits')
        if residual is None:
            residual = _read(result_dir, f'residual_{key}_d.fits')
        figures[f'fit_{index}_{key}'] = _fit_figure(
            key, data, error, model, residual, psf)
    return figures


def _psf_shape(result_dir: str) -> _EllipseShape | None:
    """
    Return the ellipse at half the maximum of the PSF (from its second
    moments), if there is one constant PSF image.
    """
    files = [f for f in os.listdir(result_dir) if _PSF_FILE.fullmatch(f)]
    image = _read(result_dir, files[0]) if len(files) == 1 else None
    if image is None or image.ndim != 2 or not image.sum() > 0:
        return None
    y, x = np.indices(image.shape)
    weights = image / image.sum()
    xc, yc = np.sum(weights * x), np.sum(weights * y)
    dx, dy = x - xc, y - yc
    cov = np.array([
        [np.sum(weights * dx * dx), np.sum(weights * dx * dy)],
        [np.sum(weights * dx * dy), np.sum(weights * dy * dy)]])
    variances, vectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    major, minor = _FWHM * np.sqrt(np.maximum(variances[::-1], 0))
    return major, minor, angle


def _fit_figure(
        key: str,
        data: np.ndarray,
        error: np.ndarray | None,
        model: np.ndarray,
        residual: np.ndarray | None,
        psf: _EllipseShape | None
) -> Figure:
    """Return the figure of the data, model and residuals of an item."""
    if data.ndim == 1:
        return _profile_figure(key, data, error, model)
    if data.ndim == 3:
        # The moment 0 of the cube, and its position-velocity slice
        # through its central row (velocity along the vertical axis)
        figure = Figure(figsize=(13, 8.5), layout='constrained')
        axes = figure.subplots(2, 3, squeeze=False)
        sums = [np.nansum(a, axis=0) for a in (data, model)]
        _map_panels(
            figure, axes[0], 'moment 0', sums[0], sums[1],
            sums[1] - sums[0], 'model - data', psf, 'equal')
        row = data.shape[1] // 2
        pv_residual = (model - data if residual is None else residual)
        _map_panels(
            figure, axes[1], f'row {row}', data[:, row, :], model[:, row, :],
            pv_residual[:, row, :],
            'model - data' if residual is None else 'residual (errors)',
            None, 'auto')
        return figure
    figure = Figure(figsize=(13, 4.2), layout='constrained')
    axes = figure.subplots(1, 3, squeeze=False)
    _map_panels(
        figure, axes[0], key, data, model,
        model - data if residual is None else residual,
        'model - data' if residual is None else 'residual (errors)', psf,
        'auto' if 'spectra' in key else 'equal')
    return figure


def _map_panels(
        figure: Figure,
        axes: Sequence[Axes],
        label: str,
        data: np.ndarray,
        model: np.ndarray,
        residual: np.ndarray,
        residual_label: str,
        psf: _EllipseShape | None,
        aspect: Literal['equal', 'auto']
) -> None:
    """Draw the data, model and residual of a map on three axes."""
    finite = data[np.isfinite(data)]
    if not finite.size:
        finite = model[np.isfinite(model)]
    low, high = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
    cmap = 'viridis'
    if label.startswith('moment1'):
        # Velocities about their median, on a diverging map
        median = np.median(finite) if finite.size else 0
        half = max(abs(low - median), abs(high - median))
        low, high, cmap = median - half, median + half, 'RdBu_r'
    finite_residual = np.abs(residual[np.isfinite(residual)])
    limit = np.percentile(finite_residual, 99) \
        if finite_residual.size else 1.0
    limit = limit if limit > 0 else 1.0
    panels = [
        (data, f'{label}: data', cmap, low, high),
        (model, f'{label}: model', cmap, low, high),
        (residual, f'{label}: {residual_label}', 'RdBu_r', -limit, limit)]
    for ax, (values, title, panel_cmap, vmin, vmax) in zip(axes, panels):
        image = ax.imshow(
            values, origin='lower', cmap=panel_cmap, vmin=vmin, vmax=vmax,
            interpolation='nearest', aspect=aspect)
        figure.colorbar(image, ax=ax, shrink=0.85)
        ax.set_title(title)
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)' if aspect == 'equal' else 'channel')
    if psf is not None:
        width, height, angle = psf
        margin = 0.75 * max(width, height) + 1
        for ax in axes[:2]:
            ax.add_patch(Ellipse(
                (margin, margin), width, height, angle=angle,
                facecolor='0.6', edgecolor='white', alpha=0.85))


def _profile_figure(
        key: str,
        data: np.ndarray,
        error: np.ndarray | None,
        model: np.ndarray
) -> Figure:
    """Return the figure of the data and model of values per region."""
    figure = Figure(figsize=(8, 6), layout='constrained')
    top, bottom = figure.subplots(
        2, 1, sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    regions = np.arange(len(data))
    top.errorbar(regions, data, yerr=error, fmt='o', label='data')
    top.plot(regions, model, '-', label='model')
    top.set_title(key)
    top.legend()
    bottom.axhline(0, color='0.5', lw=0.8)
    if error is None:
        bottom.plot(regions, model - data, 'o')
        bottom.set_ylabel('model - data')
    else:
        bottom.plot(regions, (model - data) / error, 'o')
        bottom.set_ylabel('residual (errors)')
    bottom.set_xlabel('region')
    return figure


def _posterior_figures(
        result_dir: str,
        only_best: bool,
        separate: bool,
        params: Sequence[str] | None
) -> dict[str, Figure]:
    """
    Return the corner plots of the posterior samples, or (without any) of
    the Laplace samples of each solution.
    """
    with open(os.path.join(result_dir, 'result.json')) as f:
        solutions = json.load(f)['solutions']
    names = list(params) if params else list(solutions[0]['free'])
    sample_sets = []
    if os.path.exists(os.path.join(result_dir, 'posterior.csv')):
        sample_sets.append(('posterior', 'posterior.csv', solutions[0]))
    else:
        for k, solution in enumerate(solutions[:1] if only_best
                                     else solutions):
            filename = f'solution_{k}_laplace.csv'
            if os.path.exists(os.path.join(result_dir, filename)):
                sample_sets.append((f'laplace_{k}', filename, solution))
    figures = {}
    for label, filename, solution in sample_sets:
        header, samples, weights = _read_samples(
            os.path.join(result_dir, filename))
        if missing := [n for n in names if n not in header]:
            raise ValueError(
                f"these elements are not in {filename}: {missing}; it has "
                f"{header}")
        columns = samples[:, [header.index(n) for n in names]]
        truths = [solution['elements'][n] for n in names]
        title = 'posterior' if label == 'posterior' else \
            f'solution {label.split("_")[1]}: Laplace approximation'
        if not separate:
            figures[label] = _corner(columns, weights, names, truths, title)
            continue
        for i, j in itertools.combinations(range(len(names)), 2):
            figures[f'{label}_{_safe(names[i])}_{_safe(names[j])}'] = \
                _corner(
                    columns[:, [i, j]], weights, [names[i], names[j]],
                    [truths[i], truths[j]], title)
    return figures


def _read_samples(
        filename: str
) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    """
    Return the names of the elements, the samples and their weights (if
    any) of a CSV file of samples (see FitResult.dump).
    """
    with open(filename) as f:
        header = f.readline().strip().split(',')
    samples = np.loadtxt(filename, delimiter=',', skiprows=1, ndmin=2)
    if header[-1] == 'weight':
        return header[:-1], samples[:, :-1], samples[:, -1]
    return header, samples, None


def _corner(
        samples: np.ndarray,
        weights: np.ndarray | None,
        names: list[str],
        truths: list[float],
        title: str
) -> Figure:
    """Return the corner plot of samples, with the solution marked."""
    import corner
    n = len(names)
    figure = Figure(figsize=(2.4 * n + 1, 2.4 * n + 1))
    figure.subplots(n, n)
    corner.corner(
        samples, weights=weights, labels=names, truths=truths, fig=figure,
        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.3g')
    figure.suptitle(title)
    return figure


def _safe(name: str) -> str:
    """Return the name of an element in a form fit for filenames."""
    return re.sub(r'[^\w.-]', '_', name)
