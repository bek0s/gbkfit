
from typing import Any, Literal

import dynesty
import dynesty.dynamicsampler
import numpy as np

from gbkfit.utils import funcutils, iterutils, parseutils

from .core import FitParamDynesty, FitParamsDynesty, FitterDynesty


class FitParamDynestyDNS(FitParamDynesty):
    pass


class FitParamsDynestyDNS(FitParamsDynesty):
    pass


class FitterDynestyDNS(FitterDynesty):

    @staticmethod
    def type():
        return 'dynesty.dns'

    @staticmethod
    def load_params(info, pdescs):
        return FitParamsDynestyDNS.load(info, pdescs)

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'fitter')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def __init__(
            self,
            # dynesty.dynesty.DynamicNestedSampler()
            nlive: int,
            bound:
            Literal['none', 'single', 'multi', 'balls', 'cubes'] = 'multi',
            sample:
            Literal['auto', 'unif', 'rwalk', 'slice', 'rslice', 'hslice'] = 'auto',
            update_interval: int | None = None,
            first_update: dict[str, Any] | None = None,
            seed: int = 0, # instead of rstate
            enlarge: int | float | None = None,
            bootstrap: int | None = None,
            walks: int | None = None, # only when sample = 'rwalk'
            facc: float | None = 0.5, # only when sample = 'rwalk'
            slices: int | None = None, # only for slice', 'rslice', 'hslice'
            fmove: float | None = 0.9, #only for hslice
            max_move: int | None = 100, # only for hslice
            # update_func, # no parse
            ncdim: int | None = None,
            # dynesty.dynamicsampler.DynamicSampler.run_nested()
            nlive_init: int | None = None,
            maxiter_init: int | None = None,
            maxcall_init: int | None = None,
            dlogz_init: int | float | None = 0.01,
            logl_max_init: int | float = np.inf,
            n_effective_init: int = np.inf,
            nlive_batch: int | None = None,
            # wt_function=Non # no parsee,
            # wt_kwargs=None, # no parse
            maxiter_batch: int | None = None,
            maxcall_batch: int | None = None,
            maxiter: int | None = None,
            maxcall: int | None = None,
            maxbatch: int | None = None,
            n_effective: int | None = None,
            # stop_function=None, # no parse
            # stop_kwargs=None, # no parse
            use_stop=True,
            save_bounds: bool = True,
            print_progress: bool = True,
            # print_func=None # no parse

            # todo: investigate options:
            #   dynesty.dynesty.DynamicNestedSampler():
            #     queue_size, pool, use_pool,
            #     save_history, history_filename
            #   dynesty.sampler.Sampler.run_nested():
            #     checkpoint_file, checkpoint_every, resume
    ):
        # Extract dynesty.DynamicNestedSampler() arguments



        args_factory = iterutils.extract_subdict(
            locals(), funcutils.extract_args(
                dynesty.DynamicNestedSampler)[0])
        # Extract dynesty.dynamicsampler.DynamicSampler.run_nested() arguments
        args_run_nested = iterutils.extract_subdict(
            locals(), funcutils.extract_args(
                dynesty.dynamicsampler.DynamicSampler.run_nested)[0])
        super().__init__(args_factory, args_run_nested)
