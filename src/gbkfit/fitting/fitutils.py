
import logging

import numpy as np

from gbkfit.params import parsers as param_parsers
from gbkfit.utils import iterutils


_log = logging.getLogger(__name__)


def load_params_dict(info, descs, loader):
    infos, exprs = param_parsers.parse_param_info(info, descs)[4:]
    for key, val in infos.items():
        try:
            infos[key] = loader(val)
        except Exception as e:
            raise RuntimeError(
                f"could not parse information for parameter '{key}'; "
                f"reason: {str(e)}") from e
    return infos | exprs


def dump_params_dict(parameters, type_):
    info = dict()
    for key, value in parameters.items():
        if isinstance(value, type_):
            value = value.dump()
        elif iterutils.is_sequence(value):
            value = [p.dump() if isinstance(p, type_) else p for p in value]
        info[key] = value
    return info


def prepare_param_property_prior(info):
    if (prior := info.get('prior')) is None:
        if 'min' not in info or 'max' not in info:
            raise RuntimeError(
                f"option 'prior' not found in the parameter configuration; "
                f"an attempt to generate a default uniform prior failed "
                f"because 'min' and/or 'max' options are missing")
        prior = dict(type='uniform', min=info.pop('min'), max=info.pop('max'))
    return prior


def prepare_param_initial_value_from_value_min_max(
        init_value, minimum, maximum):
    has_init_value = init_value is not None and np.isfinite(init_value)
    has_minimum = minimum is not None and np.isfinite(minimum)
    has_maximum = maximum is not None and np.isfinite(maximum)
    if not has_init_value and not (has_minimum and has_maximum):
        raise RuntimeError(
            f"no initial value for the parameter was provided; "
            f"an attempt to recover the initial value failed "
            f"because the minimum and/or maximum values are not provided")
    return init_value if has_init_value else (minimum + maximum) / 2


def prepare_param_initial_width_from_width_min_max(
        init_width, minimum, maximum):
    has_init_width = init_width is not None and np.isfinite(init_width)
    has_minimum = minimum is not None and np.isfinite(minimum)
    has_maximum = maximum is not None and np.isfinite(maximum)
    if not has_init_width and not (has_minimum and has_maximum):
        raise RuntimeError(
            f"no initial value width for the parameter was provided; "
            f"an attempt to recover the width of the initial value failed "
            f"because the minimum and/or maximum values are not provided")
    return init_width if has_init_width else maximum - minimum


def prepare_param_initial_value_range_from_value_width_min_max(
        init_value, init_width, minimum, maximum):
    init_value = prepare_param_initial_value_from_value_min_max(
        init_value, minimum, maximum)
    init_width = prepare_param_initial_width_from_width_min_max(
        init_width, minimum, maximum)
    has_minimum = minimum is not None and np.isfinite(minimum)
    has_maximum = maximum is not None and np.isfinite(maximum)
    init_value_min = init_value - init_width / 2.0
    init_value_max = init_value + init_width / 2.0
    if has_minimum:
        init_value_min = max(init_value_min, minimum)
    if has_maximum:
        init_value_max = min(init_value_max, maximum)
    return init_value, init_value_min, init_value_max


def prepare_optional_initial_value_min_max(
        initial_value, initial_width, minimum, maximum):
    if (initial_value is None) != (initial_width is None):
        _log.warning(
            f"initial_value ({initial_value}) and "
            f"initial_width ({initial_width}) will be ignored "
            f"because at least one of them is not set")
        return None, None, minimum, maximum
    initial_value_min = max(initial_value - initial_width / 2.0, minimum)
    initial_value_max = min(initial_value + initial_width / 2.0, maximum)
    return initial_value, initial_width, initial_value_min, initial_value_max


def prepare_param_property_initial_value_width(initial_value, initial_width):
    if (initial_value is None) != (initial_width is None):
        _log.warning(
            f"initial value ({initial_value}) and "
            f"initial width ({initial_width}) will be ignored "
            f"because at least one of them is not set")
        initial_value = None
        initial_width = None
    return initial_value, initial_width


def residual_scalar(eparams, parameters, objective, callback):
    params = parameters.evaluate(eparams, check=True)
    # params['xpos'] = 1
    residual = objective.residual_scalar(params, squared=False)
    # print(f"residual scalar: {residual}, params: {params}")
    if callback:
        raise NotImplementedError()
    return residual


def residual_vector(eparams, parameters, objective, callback):
    params = parameters.evaluate(eparams, check=True)
    residual = objective.residual_vector_h(params)
    residual = np.nan_to_num(np.concatenate(residual, casting='safe'))
    print(f"residual vector: {residual}, params: {params}")
    if callback:
        raise NotImplementedError()
    return residual


def log_likelihood_with_prior(eparams):
    pass


def log_likelihood_without_prior(eparams, parameters, objective):
    params = parameters.evaluate(eparams, check=True)

    foo = objective.log_likelihood(params)

    return foo[0]


def nested_sampling_prior_transform(eparams, parameters):
    return parameters.priors().rescale(eparams)


def reorder_log_likelihood(unsorted_loglikes, unsorted_samples, sorted_samples):
    idxs = []
    for i in range(len(unsorted_loglikes)):
        idx = np.where(np.all(sorted_samples[i] == unsorted_samples, axis=1))[0]
        idxs.append(idx[0])
    return unsorted_loglikes[idxs]
