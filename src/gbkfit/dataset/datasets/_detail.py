
from gbkfit.dataset.data import data_parser
from gbkfit.utils import parseutils


def load_dataset_common(cls, info, names):
    desc = parseutils.make_typed_desc(cls, 'dataset')
    opts = parseutils.parse_options_for_callable(
        info, desc, cls.__init__,
        add_optional=['step', 'rpix', 'rval', 'rota'])
    step = opts.pop('step', None)
    rpix = opts.pop('rpix', None)
    rval = opts.pop('rval', None)
    rota = opts.pop('rota', None)
    for name in names:
        if name in opts:
            opts[name] = data_parser.load(opts[name], step, rpix, rval, rota)
    return opts


def dump_dataset_common(
        dataset, prefix='', dump_full_path=True, overwrite=False):
    out = dict(type=dataset.type())
    for key, data in dataset.items():
        filename_d = f'{prefix}{key}_d.fits'
        filename_m = f'{prefix}{key}_m.fits'
        filename_e = f'{prefix}{key}_e.fits'
        out[key] = data.dump(
            filename_d, filename_m, filename_e, dump_full_path, overwrite)
    return out
