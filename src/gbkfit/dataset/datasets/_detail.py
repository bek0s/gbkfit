
from gbkfit.dataset import data_parser
from gbkfit.utils import parseutils


def load_dataset_common(cls, info, names):
    desc = parseutils.make_typed_desc(cls, 'dataset')
    opts = parseutils.parse_options_for_callable(
        info, desc, cls.__init__,
        add_optional=['step', 'rpix', 'rval', 'rota'])
    for name in names:
        if name in opts:
            opts[name] = data_parser.load_one(
                opts[name],
                opts.get('step'),
                opts.get('rpix'),
                opts.get('rval'),
                opts.get('rota'))
    opts.pop('step', None)
    opts.pop('rpix', None)
    opts.pop('rval', None)
    opts.pop('rota', None)
    return opts


def dump_dataset_common(dataset, prefix='', overwrite=False):
    out = dict(dtype=dataset.type())
    for key, data in dataset.items():
        filename_d = f'{prefix}{key}_d.fits'
        filename_m = f'{prefix}{key}_m.fits'
        filename_e = f'{prefix}{key}_e.fits'
        out[key] = data.dump(filename_d, filename_m, filename_e, overwrite)
    return out
