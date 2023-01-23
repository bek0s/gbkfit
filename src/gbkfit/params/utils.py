
from gbkfit.params.symbols import *


__all__ = [
    'sort_param_enames',
    'get_unknown_params',
    'get_missing_params',
    'ensure_no_unknown_params',
    'ensure_no_missing_params'
]


def sort_param_enames(enames, pdescs):
    # Make sure all enames are known
    if unknown := get_unknown_params(enames, pdescs):
        raise RuntimeError(f"unknown parameter enames: {unknown}")
    # Make all possible enames. We want to follow that order.
    enames_all = make_param_symbols_from_pdescs(pdescs.values(), pdescs.keys())
    # Return the supplied enames in the correct order
    return [ename for ename in enames_all if ename in enames]


def get_unknown_params(enames, pdescs):
    enames_all = make_param_symbols_from_pdescs(pdescs.values(), pdescs.keys())
    unknown = list(set(enames) - set(enames_all))
    return unknown


def get_missing_params(enames, pdescs):
    enames_all = make_param_symbols_from_pdescs(pdescs.values(), pdescs.keys())
    missing = list(set(enames_all) - set(enames))
    return sort_param_enames(missing, pdescs)


def ensure_no_unknown_params(eparams, pdescs):
    if unknown := get_unknown_params(eparams, pdescs):
        raise RuntimeError(
            f"the following parameters are not recognized: {unknown}")


def ensure_no_missing_params(eparams, pdescs):
    if missing := get_missing_params(eparams, pdescs):
        raise RuntimeError(
            f"the following parameters are missing: {missing}")
