"""
The plot task: figures of a run of a fit (or of the eval task in
objective mode): the data, model and residuals of each data item, and
corner plots of the posterior.
"""

import io
import itertools
import logging
import os
import re
from collections.abc import Sequence
from typing import Literal, NamedTuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

import gbkfit.outputs
from gbkfit.outputs import Product, Run
from . import _detail


__all__ = [
    'plot'
]


_log = logging.getLogger(__name__)

# The width of a Gaussian at half its maximum, in standard deviations
_FWHM = 2 * np.sqrt(2 * np.log(2))

# An ellipse: its width, height and angle (degrees), in pixels
_EllipseShape = tuple[float, float, float]


class _Figure(NamedTuple):
    """
    A figure, as a product of the run of the plot task: its path (without
    the extension), the ids of what it shows, and the products of the
    source run it is made from.
    """
    path: str
    figure: Figure
    ids: dict[str, str]
    sources: list[Product]


def plot(
        result_dir: str,
        output_dir: str | None,
        output_dir_mode: Literal['terminate', 'overwrite', 'unique'],
        format_: Literal['pdf', 'png'],
        dpi: int,
        only_best: bool,
        posterior_mode: Literal['none', 'corner', 'separate'],
        posterior_params: Sequence[str] | None
) -> None:
    """
    Make the figures of a run of a fit (or of the eval task in objective
    mode), as a run of their own (see gbkfit.outputs; its options are the
    source run, relative to it, and those below), of products of kind
    'figure', with the ids of what they show and the products they are
    made from as their sources:

    - for each data item of each observation, its data, model and
      residuals (observations/<observation>/fit/<item>; ids observation,
      item and plot 'fit');
    - with a fit result, corner plots of the posterior samples
      (fit/posterior; plot 'posterior'), or of the Laplace samples of the
      solutions (fit/solutions/<k>/laplace; plot 'laplace', solution k);
      one of each pair of elements is in a folder of that path, named
      after the pair (ids x and y).

    Maps are shown as images (with the PSF of the observation, if
    constant, in a corner), cubes as their moment 0 and the
    position-velocity slice through their central row, and values per
    region as lines.

    Parameters
    ----------
    result_dir : str
        The directory of the run.
    output_dir : str, optional
        The directory of the run of the figures (by default, the folder
        plots of the run).
    output_dir_mode : {'terminate', 'overwrite', 'unique'}
        What to do if the directory of the figures exists (see
        gbkfit.outputs.create).
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
        If the directory is not a run, or a posterior element is not in
        the samples.
    """
    source = gbkfit.outputs.open(result_dir)
    if output_dir is None:
        output_dir = os.path.join(source.path, 'plots')
    figures = _fit_figures(source)
    if posterior_mode != 'none':
        figures += _posterior_figures(
            source, only_best, posterior_mode == 'separate',
            posterior_params)
    if not figures:
        _log.warning(f"found nothing to plot in '{source.path}'")
    options = dict(
        source=os.path.relpath(source.path, os.path.abspath(output_dir)),
        format=format_, dpi=dpi, only_best=only_best,
        posterior_mode=posterior_mode,
        posterior_params=None if posterior_params is None
        else list(posterior_params))
    with gbkfit.outputs.create(
            output_dir, 'plot', output_dir_mode, options) as run, \
            _detail.run_log(run):
        for figure in figures:
            buffer = io.BytesIO()
            figure.figure.savefig(
                buffer, format=format_, dpi=dpi, bbox_inches='tight')
            run.write(
                f'{figure.path}.{format_}', 'figure', buffer.getvalue(),
                sources=[p.path for p in figure.sources], **figure.ids)
    _log.info(f"wrote {len(figures)} figures to '{run.path}'")


def _read(source: Run, product: Product | None) -> np.ndarray | None:
    """Return the data of a FITS product, or None without a product."""
    if product is None:
        return None
    return np.asarray(source.read(product).data, dtype=float)


def _find(source: Run, kind: str, **ids: str) -> Product | None:
    """Return the product of a kind and ids of a run, if it has one."""
    found = source.products(kind, **ids)
    return found[0] if len(found) == 1 else None


def _fit_figures(source: Run) -> list[_Figure]:
    """Return the figures of the data, model and residuals of each item."""
    figures = []
    for model in source.products('model'):
        observation, key = model.ids['observation'], model.ids['item']
        ids = dict(observation=observation, item=key)
        if (data := _find(source, 'data', **ids)) is None:
            continue
        error = _find(source, 'error', **ids)
        residual = _find(source, 'residual', **ids)
        psf = _find(source, 'extra', observation=observation, name='psf_lo')
        figure = _fit_figure(
            key, _read(source, data), _read(source, error),
            _read(source, model), _read(source, residual),
            _psf_shape(_read(source, psf)))
        figures.append(_Figure(
            f'observations/{observation}/fit/{key}', figure,
            ids | dict(plot='fit'),
            [p for p in (data, error, model, residual, psf) if p]))
    return figures


def _psf_shape(image: np.ndarray | None) -> _EllipseShape | None:
    """
    Return the ellipse at half the maximum of a PSF image (from its second
    moments), if it is one image (of a constant PSF).
    """
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
        source: Run,
        only_best: bool,
        separate: bool,
        params: Sequence[str] | None
) -> list[_Figure]:
    """
    Return the corner plots of the posterior samples of a run of a fit,
    or (without any) of the Laplace samples of each solution; none
    without a fit result.
    """
    if (result := _find(source, 'fit_result')) is None:
        return []
    solutions = source.read(result)['solutions']
    names = list(params) if params else list(solutions[0]['free'])
    sample_sets = []
    if (posterior := _find(source, 'posterior')) is not None:
        sample_sets.append((
            'fit/posterior', dict(plot='posterior'), posterior,
            solutions[0], 'posterior'))
    else:
        for k, solution in enumerate(solutions[:1] if only_best
                                     else solutions):
            laplace = _find(source, 'laplace', solution=str(k))
            if laplace is not None:
                sample_sets.append((
                    f'fit/solutions/{k}/laplace',
                    dict(plot='laplace', solution=str(k)), laplace, solution,
                    f'solution {k}: Laplace approximation'))
    figures = []
    for path, ids, product, solution, title in sample_sets:
        table = source.read(product)
        header = [n for n in table.colnames if n != 'weight']
        if missing := [n for n in names if n not in header]:
            raise ValueError(
                f"these elements are not in the samples of {product.path}: "
                f"{missing}; it has {header}")
        columns = np.column_stack([np.asarray(table[n]) for n in names])
        weights = np.asarray(table['weight']) \
            if 'weight' in table.colnames else None
        truths = [solution['elements'][n] for n in names]
        sources = [result, product]
        if not separate:
            figures.append(_Figure(
                path, _corner(columns, weights, names, truths, title), ids,
                sources))
            continue
        for i, j in itertools.combinations(range(len(names)), 2):
            figures.append(_Figure(
                f'{path}/{_safe(names[i])}_{_safe(names[j])}',
                _corner(
                    columns[:, [i, j]], weights, [names[i], names[j]],
                    [truths[i], truths[j]], title),
                ids | dict(x=names[i], y=names[j]), sources))
    return figures


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
