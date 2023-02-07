
import logging
import typing

import numpy as np

import gbkfit.psflsf
from gbkfit.utils import iterutils, parseutils


__init__ = [
    'load_dmodel_common',
    'dump_dmodel_common'
]


_log = logging.getLogger(__name__)


def _sanitize_dimensional_option(option, value, lengths, type_):
    args = typing.get_args(type_)
    types_ = args if args else [type_]
    type_name = " | ".join([t.__name__ for t in types_])
    lengths = iterutils.listify(lengths)
    max_length = max(lengths)
    if isinstance(value, type_):
        return iterutils.make_list(max_length, value)
    if iterutils.is_sequence_of_type(value, type_):
        if len(value) < max_length:
            raise RuntimeError(
                f"option '{option}' has a value "
                f"with a length shorter than expected; "
                f"expected length: {' or '.join(map(str, lengths))}, "
                f"current length: {len(value)}")
        if len(value) > max_length:
            new_value = value[:max_length]
            _log.warning(
                f"option '{option}' has a value "
                f"with a length longer than expected; "
                f"current length: {len(value)}; "
                f"expected length: {' or '.join(map(str, lengths))}, "
                f"the value will be trimmed from {value} to {new_value}")
            value = new_value
        return value
    raise RuntimeError(
        f"option '{option}' should be a scalar of type {type_name}, "
        f"or a sequence of type {type_name} and "
        f"length of {' or '.join(map(str, lengths))}")


def load_dmodel_common(
        cls, info, ndim, has_psf, has_lsf, dataset, expected_dataset_cls):
    desc = parseutils.make_typed_desc(cls, 'dmodel')
    # Load psf/lsf
    if has_psf and 'psf' in info:
        info['psf'] = gbkfit.psflsf.psf_parser.load_one(info['psf'])
    if has_lsf and 'lsf' in info:
        info['lsf'] = gbkfit.psflsf.lsf_parser.load_one(info['lsf'])
    # ...
    if 'dtype' in info:
        info['dtype'] = np.dtype(info['dtype']).type
    # Try to get information from the supplied dataset (optional)
    if dataset:
        if not isinstance(dataset, expected_dataset_cls):
            expected_dataset_type_desc = parseutils.make_typed_desc(
                expected_dataset_cls, 'dataset')
            provided_dataset_type_desc = parseutils.make_typed_desc(
                dataset.__class__, 'dataset')
            raise RuntimeError(
                f"{desc} is not compatible with the supplied dataset "
                f"and cannot be used to describe its properties; "
                f"expected dataset type: {expected_dataset_type_desc}; "
                f"provided dataset type: {provided_dataset_type_desc}")
        info.update(dict(
            size=dataset.size(),
            step=info.get('step', dataset.step()),
            rpix=info.get('rpix', dataset.rpix()),
            rval=info.get('rval', dataset.rval()),
            rota=info.get('rota', dataset.rota()),
            dtype=info.get('dtype', str(dataset.dtype))))
    # Validate, sanitize, and prepare dimensional options.
    # While we could rely on the type hint validation of
    # parseutils.parse_options_for_callable or other assertions inside
    # the __init__() method, we prepare those options here. This allows
    # us to be more tolerant with their values, when possible.
    # For example, we can now convert an scube configuration to
    # an image configuration by just changing its type, without
    # having to adjust any dimensional options. The code will just use
    # the first two dimensions and ignore the last one.
    for option_name, option_type in [
            ('size', int),
            ('step', int | float),
            ('rpix', int | float),
            ('rval', int | float),
            ('scale', int)]:
        if option_name in info:
            info[option_name] = _sanitize_dimensional_option(
                option_name, info[option_name], ndim, option_type)
    # Parse options and create object
    opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
    return opts


def dump_dmodel_common(dmodel):
    info = dict(
        type=dmodel.type(),
        size=dmodel.size(),
        step=dmodel.step(),
        cval=dmodel.cval(),
        rota=dmodel.rota(),
        scale=dmodel.scale(),
        dtype=dmodel.dtype())
    if hasattr(dmodel, 'psf'):
        info.update(psf=gbkfit.psflsf.psf_parser.dump(dmodel.psf()))
    if hasattr(dmodel, 'lsf'):
        info.update(lsf=gbkfit.psflsf.lsf_parser.dump(dmodel.lsf()))
    return info
