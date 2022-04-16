
from gbkfit.params.descs import ParamScalarDesc, ParamVectorDesc
from gbkfit.utils import miscutils, parseutils
from .core import *


def explode_param_name_from_indices(name, indices):
    eparams = []
    for index in iterutils.listify(indices):
        eparams.append(make_param_symbol(name, index))
    return eparams


def explode_param_names_from_indices(name_list, indices_list):
    # todo: add strict=True to zip() (python 3.10)
    eparams = []
    for name, indices in zip(name_list, indices_list):
        eparams.extend(explode_param_name_from_indices(name, indices))
    return eparams


def explode_param_name_from_desc(desc, override_name=None):
    enames = []
    name = override_name if override_name else desc.name()
    if isinstance(desc, ParamScalarDesc):
        enames.extend(explode_param_name_from_indices(name, None))
    elif isinstance(desc, ParamVectorDesc):
        enames.extend(explode_param_name_from_indices(
            name, list(range(desc.size()))))
    else:
        raise RuntimeError()
    return enames


def explode_param_names_from_descs(descs, override_names=None):
    # todo: add strict=True to zip() (python 3.10)
    enames = []
    names = override_names if override_names else [d.name() for d in descs]
    for name, desc in zip(names, descs):
        enames.extend(explode_param_name_from_desc(desc, name))
    return enames


def sort_param_enames(descs, enames):
    # todo: check if enames is indeed a subset?
    enames_all = explode_param_names_from_descs(descs.values(), descs.keys())
    return [ename for ename in enames_all if ename in enames]
