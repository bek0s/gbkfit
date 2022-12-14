#
# import numpy as np
#
# from gbkfit.math import *
#
#
# def test_foo(capsys):
#
#     import bilby.core
#
#     with capsys.disabled():
#
#         assert gauss_1d_cdf(-np.inf, 0,  1) == 0
#         assert gauss_1d_cdf(+np.inf, 0, 1) == 1
#         assert gauss_1d_cdf(0.0, 0, 1) == 0.5
#
#         xmin = -20
#         xmax = +20
#         mean = 2.0
#         scale = 8.0
#         tmin = -1.5 * scale
#         tmax = +1.5 * scale
#
#         xdata = np.arange(xmin, xmax, 0.01)
#         ydata = np.arange(0.0, 1.0, 0.01)
#
#         fun1 = gauss_trunc_1d_fun(xdata, 1, mean, scale, tmin, tmax)
#         pdf1 = gauss_trunc_1d_pdf(xdata, mean, scale, tmin, tmax)
#         cdf1 = gauss_trunc_1d_cdf(xdata, mean, scale, tmin, tmax)
#         ppf1 = gauss_trunc_1d_ppf(ydata, mean, scale, tmin, tmax)
#
#         p = bilby.core.prior.TruncatedGaussian(mean, scale, tmin, tmax)
#         pdf2 = p.prob(xdata)
#         cdf2 = p.cdf(xdata)
#         ppf2 = p.rescale(ydata)
#
#
#         import matplotlib.pyplot as plt
#         alpha1 = 0.2
#         color1 = 'tab:blue'
#         color2 = 'tab:orange'
#         color3 = 'tab:gray'
#         color4 = 'tab:pink'
#         style1 = '-'
#         style2 = ':'
#         style3 = '--'
#         style4 = '--'
#         fig = plt.figure(figsize=(4, 10), dpi=200)
#
#         ax = fig.add_subplot(4, 1, 1)
#         ax.set_xlim(1.2 * xmin, 1.2 * xmax)
#         ax.axvline(x=-scale, linestyle=style3, color=color3, alpha=alpha1)
#         ax.axvline(x=+scale, linestyle=style3, color=color3, alpha=alpha1)
#         ax.axvline(x=tmin, linestyle=style4, color=color4, alpha=alpha1)
#         ax.axvline(x=tmax, linestyle=style4, color=color4, alpha=alpha1)
#         ax.plot(xdata, fun1, linestyle=style1, color=color1)
#         ax.plot(xdata, fun1, linestyle=style2, color=color2)
#
#         ax = fig.add_subplot(4, 1, 2)
#         ax.set_xlim(1.2 * xmin, 1.2 * xmax)
#         ax.axvline(x=-scale, linestyle=style3, color=color3, alpha=alpha1)
#         ax.axvline(x=+scale, linestyle=style3, color=color3, alpha=alpha1)
#         ax.axvline(x=tmin, linestyle=style4, color=color4, alpha=alpha1)
#         ax.axvline(x=tmax, linestyle=style4, color=color4, alpha=alpha1)
#         ax.plot(xdata, pdf1, linestyle=style1, color=color1)
#         ax.plot(xdata, pdf2, linestyle=style2, color=color2)
#         ax = fig.add_subplot(4, 1, 3)
#         ax.set_xlim(1.2 * xmin, 1.2 * xmax)
#         ax.set_ylim(-0.05, 1.05)
#         ax.axvline(x=-scale, linestyle=style3, color=color3, alpha=alpha1)
#         ax.axvline(x=+scale, linestyle=style3, color=color3, alpha=alpha1)
#         ax.axvline(x=tmin, linestyle=style4, color=color4, alpha=alpha1)
#         ax.axvline(x=tmax, linestyle=style4, color=color4, alpha=alpha1)
#         ax.plot(xdata, cdf1, linestyle=style1, color=color1)
#         ax.plot(xdata, cdf2, linestyle=style2, color=color2)
#         ax = fig.add_subplot(4, 1, 4)
#         ax.set_xlim(1.2 * xmin, 1.2 * xmax)
#         ax.set_ylim(-0.05, 1.05)
#         ax.axvline(x=-scale, linestyle=style3, color=color3, alpha=alpha1)
#         ax.axvline(x=+scale, linestyle=style3, color=color3, alpha=alpha1)
#         ax.axvline(x=tmin, linestyle=style4, color=color4, alpha=alpha1)
#         ax.axvline(x=tmax, linestyle=style4, color=color4, alpha=alpha1)
#         ax.plot(ppf1, ydata, linestyle=style1, color=color1)
#         ax.plot(ppf2, ydata, linestyle=style2, color=color2)
#
#         plt.show()
#
#     pass
