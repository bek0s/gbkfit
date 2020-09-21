
import gbkfit.dataset
from gbkfit.utils import parseutils


def load_dataset_common(cls, info, names):
    desc = gbkfit.dataset.make_dataset_desc(cls)
    opts = parseutils.parse_options_for_callable(
        info, desc, cls.__init__,
        add_optional=['step', 'rpix', 'rval', 'rota'])
    for name in names:
        if name in opts:
            opts[name] = gbkfit.dataset.data_parser.load_one(
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


def dump_dataset_common(dataset, prefix=''):
    out = dict(dtype=dataset.type())
    for key, data in dataset.items():
        out[key] = data.dump(f"{prefix}{key}_")
    return out
