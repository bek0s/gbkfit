from gbkfit.fitting.base import fitter_parser


# The fitters, imported when first asked for: most need optional packages
fitter_parser.register_optional({
    'lmfit.least_squares':
        'gbkfit.fitting.fitters.lmfit.FitterLMFitLeastSquares',
    'scipy.least_squares':
        'gbkfit.fitting.fitters.scipy.FitterScipyLeastSquares',
    'multistart': 'gbkfit.fitting.fitters.multistart.FitterMultiStart',
    'cmaes': 'gbkfit.fitting.fitters.cmaes.FitterCMAES',
    'emcee': 'gbkfit.fitting.fitters.emcee.FitterEmcee',
    'zeus': 'gbkfit.fitting.fitters.zeus.FitterZeus',
    'nautilus': 'gbkfit.fitting.fitters.nautilus.FitterNautilus',
    'dynesty.static': 'gbkfit.fitting.fitters.dynesty.FitterDynestyStatic',
    'dynesty.dynamic': 'gbkfit.fitting.fitters.dynesty.FitterDynestyDynamic',
    'ultranest': 'gbkfit.fitting.fitters.ultranest.FitterUltraNest',
})
