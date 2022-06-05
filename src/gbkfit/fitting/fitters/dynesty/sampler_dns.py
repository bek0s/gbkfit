
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
            # dynesty.DynamicNestedSampler() arguments
            bound='multi',
            sample='auto',
            update_interval=None,
            first_update=None,
            rstate=None,
            enlarge=None,
            bootstrap=0,
            walks=25,
            facc=0.5,
            slices=None,
            fmove=0.9,
            max_move=100,
            # dynesty.dynamicsampler.DynamicSampler.run_nested() arguments
            nlive_init=500,
            maxiter_init=None,
            maxcall_init=None,
            dlogz_init=0.01,
            logl_max_init=np.inf,
            n_effective_init=np.inf,
            nlive_batch=500,
            wt_function=None,
            wt_kwargs=None,
            maxiter_batch=None,
            maxcall_batch=None,
            maxiter=None,
            maxcall=None,
            maxbatch=None,
            n_effective=None,
            stop_function=None,
            stop_kwargs=None,
            use_stop=True,
            save_bounds=True,
            print_progress=True,
            print_func=None):
        # Extract dynesty.DynamicNestedSampler() arguments
        args_factory = iterutils.extract_subdict(
            locals(), funcutils.extract_args(
                dynesty.DynamicNestedSampler)[0])
        # Extract dynesty.dynamicsampler.DynamicSampler.run_nested() arguments
        args_run_nested = iterutils.extract_subdict(
            locals(), funcutils.extract_args(
                dynesty.dynamicsampler.DynamicSampler.run_nested)[0])
        super().__init__(args_factory, args_run_nested)
