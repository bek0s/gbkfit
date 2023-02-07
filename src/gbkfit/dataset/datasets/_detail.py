
import logging
import typing

from gbkfit.dataset.data import data_parser
from gbkfit.utils import iterutils, parseutils


__init__ = [
    'load_dataset_common',
    'dump_dataset_common'
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


def load_dataset_common(cls, info, names, ndim, **kwargs):
    prefix = kwargs.get('prefix', '')
    desc = parseutils.make_typed_desc(cls, 'dataset')
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
            ('rval', int | float)]:
        if option_name in info:
            info[option_name] = _sanitize_dimensional_option(
                option_name, info[option_name], ndim, option_type)
    # Read global coordinate system options.
    # These will apply to all data in the dataset that do not define
    # their own options.
    step = info.pop('step', None)
    rpix = info.pop('rpix', None)
    rval = info.pop('rval', None)
    rota = info.pop('rota', None)
    # Load all data in the dataset, using the above options
    for name in names:
        if name in info:
            info[name] = data_parser.load(
                info[name], step, rpix, rval, rota, prefix)
    # Parse options and return them
    opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
    return opts


def dump_dataset_common(dataset, **kwargs):
    prefix = kwargs.get('prefix', '')
    dump_path = kwargs.get('dump_path', True)
    overwrite = kwargs.get('overwrite', False)
    info = dict(type=dataset.type())
    info.update(
        step=dataset.step(),
        rpix=dataset.rpix(),
        rval=dataset.rval(),
        rota=dataset.rota())
    for key, data in dataset.items():
        filename_d = f'{prefix}{key}_d.fits'
        filename_m = f'{prefix}{key}_m.fits'
        filename_e = f'{prefix}{key}_e.fits'
        info[key] = data.dump(
            filename_d, filename_m, filename_e,
            dump_wcs=False,  # Reduce unnecessary verbosity
            dump_path=dump_path, overwrite=overwrite)
    return info
