

from gbkfit.dataset.data import data_parser
from gbkfit.utils import parseutils


__all__ = [
    'load_dataset_common',
    'dump_dataset_common'
]


def load_dataset_common(cls, info, names, **kwargs):
    """
    The options of a dataset of class cls: its data items (names)
    loaded.
    """
    prefix = kwargs.get('prefix', '')
    desc = parseutils.make_typed_desc(cls, 'dataset')
    parseutils.sanitize_dimensional_options(info, dict(
        size=int, step=int | float, rpix=int | float, rval=int | float),
        cls._ndim)
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
            with parseutils.config_path(name):
                info[name] = data_parser.load(
                    info[name], step, rpix, rval, rota, prefix,
                    spectral_axis=cls._spectral_axis)
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
