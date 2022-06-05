
import logging.config
import os.path

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np

import gbkfit.fitting.result

from gbkfit.utils import miscutils


_log = logging.getLogger(__name__)

# Enable LaTeX
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

#
# Various constants for controlling figure visuals
#

FIG_SCALE = 1.0

DMR_FIG_SIZE_X = FIG_SCALE * 10
DMR_FIG_SIZE_Y = FIG_SCALE * 3

FIG_PLOT_TITLE_FONT_SIZE = FIG_SCALE * 16

FIG_PLOT_XLABEL_FONT_SIZE = FIG_SCALE * 16
FIG_PLOT_YLABEL_FONT_SIZE = FIG_SCALE * 16

FIG_PLOT_ID_FONT_SIZE = FIG_SCALE * 16
FIG_PLOT_ID_LINEWIDTH = FIG_SCALE * 2
FIG_PLOT_ID_COLOR = 'white'
FIG_PLOT_ID_POS_X = 1
FIG_PLOT_ID_POS_Y = 1

FIG_OBJ_ID_FONT_SIZE = FIG_SCALE * 16
FIG_OBJ_ID_LINEWIDTH = FIG_SCALE * 2
FIG_OBJ_ID_COLOR = 'white'
FIG_OBJ_ID_POS_X = 1
FIG_OBJ_ID_POS_Y = 1

FIG_PSF_COLOR = 'gray'
FIG_PSF_ALPHA = 0.8
FIG_PSF_MARGIN_X = 10
FIG_PSF_MARGIN_Y = 10


MMAP_COLORMAP_NAMES = ['RdBu_r', 'RdBu_r', 'RdBu_r']


def setup_plot_text(
        ax, label, posx, posy, fontsize, linewidth, foreground, halign, valign):
    patheffect = patheffects.withStroke(
        linewidth=linewidth, foreground=foreground)
    ax.text(
        posx, posy, label, transform=ax.transAxes, fontsize=fontsize,
        horizontalalignment=halign, verticalalignment=valign,
        patheffects=[patheffect])


def setup_plot_id(ax, label):
    setup_plot_text(
        ax, label, FIG_PLOT_ID_POS_X, FIG_PLOT_ID_POS_Y, FIG_PLOT_ID_FONT_SIZE,
        FIG_PLOT_ID_LINEWIDTH, FIG_PLOT_ID_COLOR, 'right', 'top')


def setup_object_id(ax, label):
    setup_plot_text(
        ax, label, FIG_OBJ_ID_POS_X, FIG_OBJ_ID_POS_Y, FIG_OBJ_ID_FONT_SIZE,
        FIG_OBJ_ID_LINEWIDTH, FIG_OBJ_ID_COLOR, 'left', 'top')


def setup_psf(ax, size, angle):
    pos = (FIG_PSF_MARGIN_X + size[0] / 2, FIG_PSF_MARGIN_Y + size[1] / 2)
    ax.add_artist(patches.Ellipse(
        pos, size[0], size[1], angle=angle,
        facecolor=FIG_PSF_COLOR, alpha=FIG_PSF_ALPHA))


def setup_title(ax, label):
    pass


def setup_image(ax, data, cmap, vmin=None, vmax=None):
    return ax.imshow(
        data, cmap=cmap, vmin=vmin, vmax=vmax,
        aspect='equal', origin='lower', interpolation='foo')


def plot_best_fit_dmr(
        fig, ax_dat, ax_mdl, ax_res, data_dat, data_msk, data_mdl, data_res):

    setup_image(ax_dat, data_dat, 'foo')
    setup_image(ax_mdl, data_mdl, 'foo')
    setup_image(ax_res, data_res, 'foo')


def make_best_fit_dmr_gs(nrows, wspace, hspace, hscale, wscale):
    return gridspec.GridSpec(
        nrows, 6, wspace=wspace, hspace=hspace,
        height_ratios=[hscale], width_ratios=[1, wscale, wscale, 1, wscale, 1])


def plot_best_fit_image(dataset, solution, dpi):
    nrows = 1
    fig = plt.figure()
    gs = make_best_fit_dmr_gs(nrows, 0.04, 0.04, 15, 15)

    return dict(dmr=fig)


def plot_best_fit_lslit(dataset, solution):
    return 1


def plot_best_fit_mmaps(dataset, solution):
    nrows = len(dataset)
    fig = plt.figure()
    gs = make_best_fit_dmr_gs(nrows, 0.04, 0.04, 15, 15)

    for i, (name, data) in enumerate(dataset.items()):

        data_dat = data.data()
        data_msk = data.mask()
        data_mdl = solution['model']
        data_res = solution['resid']

        ax0 = fig.add_subplot(gs[i, 0])
        ax1 = fig.add_subplot(gs[i, 1])
        ax2 = fig.add_subplot(gs[i, 2])
        ax3 = fig.add_subplot(gs[i, 4])
        ax4 = fig.add_subplot(gs[i, 5])
        plot_best_fit_dmr(fig, ax1, ax2, ax4, None, None, None)

    return dict(dmr=fig)


def plot_best_fit_scube(dataset, solution):
    return 1


def plot_best_fit(result, solution):
    plot_fun_selector = dict(
        image=plot_best_fit_image,
        lslit=plot_best_fit_lslit,
        mmaps=plot_best_fit_mmaps,
        scube=plot_best_fit_scube)
    figures = dict()
    for i, dataset in enumerate(result.datasets):
        figs = plot_fun_selector[dataset.type()](dataset, solution)
        figures |= {f'best_fit_{i}_{k}': v for k, v in figs.items()}
    return figures


def plot_posterior(result, posterior, posterior_params):
    return dict()


def plot_solution(result, solution, posterior_params, skip_posterior):

    # Create best fit figures
    figures = plot_best_fit(result, solution)

    # Create solution posterior figures
    if solution.postrerior and not skip_posterior:
        figures |= plot_posterior(result, solution.posterior, posterior_params)

    return figures


def plot(
        result_dir, output_dir, format_, dpi,
        only_champion, posterior_params, skip_posterior):

    result_dir = os.path.abspath(result_dir)

    output_dir = os.path.abspath(miscutils.make_unique_path('figures'))

    os.makedirs(output_dir)

    _log.info(f"reading result directory: '{result_dir}'...")

    result = gbkfit.fitting.result.load_result(result_dir)

    # Decide which solutions to plot
    solutions = result.solutions

    figures = dict()

    # Create solution figures
    for i, solution in enumerate(solutions):
        figures |= plot_solution(
            result, solution, posterior_params, skip_posterior)

    # Create global posterior figures
    if result.posterior and not skip_posterior:
        figures |= plot_posterior(result, result.posterior, posterior_params)

    # Save figures
    for key, figure in figures.items():
        figure.savefig(f'figure_{key}.{format_}', bbox_inches='tight', dpi=dpi)
