
import collections.abc
import copy
from typing import Any, Literal

import dynesty
import dynesty.sampler
import dynesty.utils
import numpy as np

from gbkfit.fitting import fitutils
from gbkfit.fitting.prior import prior_parser, PriorDict
from gbkfit.fitting.result import make_fitter_result
from gbkfit.params import parsers as param_parsers
from gbkfit.utils import funcutils, iterutils, parseutils

from .core import FitParamDynesty, FitParamsDynesty, FitterDynesty, \
    log_likelihood, prior_transform


__all__ = [
    'FitParamDynestySNS',
    'FitParamsDynestySNS',
    'FitterDynestySNS'
]


class FittingParamPropertyDynestySNS(FitParamDynesty):

    @classmethod
    def load(cls, info):
        info['prior'] = fitutils.prepare_param_info_prior(info)
        info['prior'] = prior_parser.load(info['prior'], param_info=info)
        desc = parseutils.make_basic_desc(cls, 'fit parameter')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        return dict(
            prior=self.prior().dump(),
            boundary=self.boundary())

    def __init__(self, prior, boundary=None):
        super().__init__()
        boundary = boundary.lower() if isinstance(boundary, str) else boundary
        if boundary is not None and boundary not in ['periodic', 'reflective']:
            raise RuntimeError()
        self._prior = copy.deepcopy(prior)
        self._boundary = boundary

    def prior(self):
        return self._prior

    def boundary(self):
        return self._boundary


class FitParamsDynestySNS(FitParamsDynesty):

    @staticmethod
    def load_param(info):
        return FitParamDynestySNS.load(info)

    @classmethod
    def load(cls, info, pdescs):
        desc = parseutils.make_basic_desc(cls, 'fit params')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['pdescs'])
        opts = param_parsers.load_params_parameters_conversions(
            opts, pdescs, collections.abc.Mapping, cls.load_param)
        return cls(pdescs, **opts)

    def dump(self):
        info = dict()
        for k, v in self.parameters().items():
            info[k] = v.dump() if isinstance(v, FitParamDynestySNS) else v
        return info

    def __init__(self, pdescs, properties, value_conversions=None):
        super().__init__(pdescs, properties, value_conversions,
                         FitParamDynestySNS)

        self._priors = PriorDict(
            {k: v.prior() for k, v in self.infos().items()})

    def priors(self):
        return self._priors


class FitterDynestySNS(FitterDynesty):

    @staticmethod
    def type():
        return 'dynesty.sns'

    @staticmethod
    def load_params(info, descs):
        return FitParamsDynestySNS.load(info, descs)

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'fitter')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        return dict(error='error')

    def __init__(
            self,
            # dynesty.dynesty.NestedSampler()
            nlive: int,
            bound: Literal[
                'none', 'single', 'multi', 'balls', 'cubes'] = 'multi',
            sample:
            Literal[
                'auto', 'unif', 'rwalk', 'slice', 'rslice', 'hslice'] = 'auto',
            update_interval: int | None = None,
            first_update: dict[str, Any] | None = None,
            seed: int = 0,
            enlarge: int | float | None = None,
            bootstrap: int | None = None,
            walks: int | None = None,
            facc: int | float | None = 0.5,
            slices: int | None = None,
            fmove: int | float | None = 0.9,
            max_move: int | None = 100,
            ncdim: int | None = None,
            # dynesty.sampler.Sampler.run_nested()
            maxiter: int | None = None,
            maxcall: int | None = None,
            dlogz: float | None = None,
            logl_max: float = np.inf,
            n_effective: int | None = None,
            add_live: bool = True,
            print_progress: bool = True,
            save_bounds: bool = True
    ):
        locals_ = copy.deepcopy(locals())
        constructor_args = funcutils.extract_args(
            dynesty.NestedSampler)[0]
        run_nested_args = funcutils.extract_args(
            dynesty.sampler.Sampler.run_nested)[0]
        constructor_args_found = (
            iterutils.extract_sublist(constructor_args, locals_))[0]
        run_nested_args_found = (
            iterutils.extract_sublist(run_nested_args, locals_))[0]
        super().__init__(constructor_args, run_nested_args_found)

    def _fit_impl(self, objective, parameters):
        ndim = len(parameters.infos())
        # Setup boundary iterables
        boundary_periodic = []
        boundary_reflective = []
        for i, param in enumerate(parameters.infos().values()):
            match param.boundary():
                case 'periodic':
                    boundary_periodic.append(i)
                case 'reflective':
                    boundary_reflective.append(i)
        if not boundary_periodic:
            boundary_periodic = None
        if not boundary_reflective:
            boundary_reflective = None
        # Create sampler
        sampler = dynesty.NestedSampler(
            log_likelihood, prior_transform, ndim,
            logl_args=(parameters, objective), ptform_args=(parameters,),
            periodic=boundary_periodic, reflective=boundary_reflective,
            **self._options_init)
        # Run sampling
        sampler.run_nested(**self._options_run_nested)
        res = sampler.results
        print(res.summary())
        # Generate equally-weighted samples
        samples_weights = np.exp(res.logwt - res.logz[-1])
        samples_weighted = res.samples
        samples_unweighted = dynesty.utils.resample_equal(
            samples_weighted, samples_weights)
        loglikes = fitutils.reorder_log_likelihood(
            res.logl, samples_weighted, samples_unweighted)
        # ...
        posterior = dict(samples=samples_unweighted, loglikes=loglikes)
        # Extract additional information
        extra = iterutils.nativify(dict(
            nlive=res.nlive,
            niter=res.niter,
            efficiency=res.eff,
            log_evidence=res.logz[-1],
            log_evidence_err=res.logzerr[-1],
            information_gain=res.information[-1]))

        result = make_fitter_result(
            objective, parameters, posterior, solutions=(), extra=extra)

        exit()

        return result
