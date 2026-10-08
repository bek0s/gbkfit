
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import gbkfit.math


AMPLITUDE = 1.0
POSITION = 0.0
SCALE = 2.0
TRUNC_MIN = -np.inf  # -SCALE
TRUNC_MAX = +SCALE

XDATA_MIN = -11
XDATA_MAX = +11
XDATA_STEP = 0.01
XDATA = np.arange(XDATA_MIN, XDATA_MAX, XDATA_STEP)

UDATA_MIN = 0.0001
UDATA_MAX = 0.9990
UDATA_STEP = 0.01
UDATA = np.arange(UDATA_MIN, UDATA_MAX, UDATA_STEP)

FIG_DPI = 200
FIG_SIZE = (6, 8)

COLOR1 = '#e66101'
COLOR2 = '#fdb863'
COLOR3 = '#b2abd2'
COLOR4 = '#5e3c99'


FUNCTIONS = dict(
    uniform_1d=dict(
        ampl=AMPLITUDE,
        args=(POSITION - SCALE, POSITION + SCALE),
    ),
    # expon_1d=dict(
    #     ampl=AMPLITUDE,
    #     args=(POSITION, SCALE)
    # ),
    # expon_trunc_1d=dict(
    #     ampl=AMPLITUDE,
    #     args=(POSITION, SCALE, TRUNC_MIN, TRUNC_MAX)
    # ),
    # laplace_1d=dict(
    #     ampl=AMPLITUDE,
    #     args=(POSITION, SCALE)
    # ),
    # laplace_trunc_1d=dict(
    #     ampl=AMPLITUDE,
    #     args=(POSITION, SCALE, TRUNC_MIN, TRUNC_MAX)
    # ),
    # gauss_1d=dict(
    #     ampl=AMPLITUDE,
    #     args=(POSITION, SCALE)
    # ),
    # gauss_trunc_1d=dict(
    #     ampl=AMPLITUDE,
    #     args=(POSITION, SCALE, TRUNC_MIN, TRUNC_MAX)
    # ),
    # ggauss_1d=dict(
    #     ampl=AMPLITUDE,
    #     args=(POSITION, SCALE, 2.0)
    # ),
    # ggauss_trunc_1d=dict(
    #     ampl=AMPLITUDE,
    #     args=(POSITION, SCALE, 2.0, TRUNC_MIN, TRUNC_MAX)
    # ),
    # lorentz_1d=dict(
    #     ampl=AMPLITUDE,
    #     args=(POSITION, SCALE)
    # ),
    # lorentz_trunc_1d=dict(
    #     ampl=AMPLITUDE,
    #     args=(POSITION, SCALE, TRUNC_MIN, TRUNC_MAX)
    # ),
    # moffat_1d=dict(
    #     ampl=AMPLITUDE,
    #     args=(POSITION, SCALE, 2.0)
    # ),
    # moffat_trunc_1d=dict(
    #     ampl=AMPLITUDE,
    #     args=(POSITION, SCALE, 2.0, TRUNC_MIN, TRUNC_MAX)
    # ),
    # sech2_1d=dict(
    #     ampl=AMPLITUDE,
    #     args=(POSITION, SCALE)
    # ),
    # sech2_trunc_1d=dict(
    #     ampl=AMPLITUDE,
    #     args=(POSITION, SCALE, TRUNC_MIN, TRUNC_MAX)
    # )
)

for key, value in FUNCTIONS.items():

    def get_function(name, postfix):
        return getattr(gbkfit.math, f'{name}_{postfix}')


    def call_function(fun, xdata, *args):
        ydata = np.full_like(xdata, np.nan)
        try:
            ydata = fun(xdata, *args)
        except NotImplementedError:
            pass
        return ydata

    fun = get_function(key, 'fun')
    pdf = get_function(key, 'pdf')
    cdf = get_function(key, 'cdf')
    ppf = get_function(key, 'ppf')
    fun_ydata = call_function(fun, XDATA, value['ampl'], *value['args'])
    pdf_ydata = call_function(pdf, XDATA, *value['args'])
    cdf_ydata = call_function(cdf, XDATA, *value['args'])
    ppf_ydata = call_function(ppf, UDATA, *value['args'])

    def setup_axes_common(ax):
        ax.set_xlim(XDATA_MIN, XDATA_MAX)
        ax.axvline(x=-SCALE, linestyle='--', color='tab:gray', alpha=0.2)
        ax.axvline(x=+SCALE, linestyle='--', color='tab:gray', alpha=0.2)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)

    ax1 = fig.add_subplot(4, 1, 1)
    ax1.set_title(key)
    ax1.set_ylabel('FUN')
    ax1.tick_params(labelbottom=False)
    setup_axes_common(ax1)
    ax1.plot(XDATA, fun_ydata, color=COLOR1)

    ax2 = fig.add_subplot(4, 1, 2)
    ax2.set_ylabel('PDF')
    ax2.tick_params(labelbottom=False)
    setup_axes_common(ax2)
    ax2.plot(XDATA, pdf_ydata, color=COLOR2)

    ax3 = fig.add_subplot(4, 1, 3)
    ax3.set_ylabel('CDF')
    ax3.set_ylim(-0.05, 1.05)
    ax3.tick_params(labelbottom=False)
    setup_axes_common(ax3)
    ax3.plot(XDATA, cdf_ydata, color=COLOR3)
    ax3.plot(ppf_ydata, UDATA, color=COLOR4, linestyle=':')

    ax4 = fig.add_subplot(4, 1, 4)
    ax4.set_ylabel('PPF (transposed)')
    ax4.set_ylim(-0.05, 1.05)
    ax4.tick_params(labelbottom=True)
    setup_axes_common(ax4)
    ax4.plot(ppf_ydata, UDATA, color=COLOR4)

    fig.subplots_adjust(wspace=0.0, hspace=0.1)


plt.show()
