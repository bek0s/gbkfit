
import gbkfit.dataset
from gbkfit.utils import parseutils


def load_dataset_common(cls, info, names):
    desc = gbkfit.dataset.make_dataset_desc(cls)
    opts = parseutils.parse_options(
        info, desc, add_optional=['step', 'rpix', 'rval', 'rota'],
        fun=cls.__init__)[0]
    for name in names:
        if name in opts:
            opts[name] = gbkfit.dataset.data_parser.load_one(
                opts[name],
                opts.get('step'),
                opts.get('rpix'),
                opts.get('rval'),
                opts.get('rota'))
    return opts


def dump_dataset_common(dataset, prefix=''):
    out = dict(dtype=dataset.type())
    out.update({name: data.dump(prefix) for name, data in dataset.items()})
    return out
