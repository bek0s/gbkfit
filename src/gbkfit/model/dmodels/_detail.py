
import copy

import numpy as np

import gbkfit.psflsf
from gbkfit.utils import parseutils


def load_dmodel_common(cls, info, ndim, dataset, dataset_cls):
    desc = parseutils.make_typed_desc(cls, 'dmodel')
    info = copy.deepcopy(info)
    if dataset:
        dataset_desc = parseutils.make_typed_desc(dataset_cls, 'dataset')
        if not isinstance(dataset, dataset_cls):
            raise RuntimeError(
                f"{desc} "
                f"is not compatible with "
                f"{dataset_desc}")
        info.update(dict(
            size=dataset.size(),
            step=info.get('step', dataset.step()),
            rpix=info.get('rpix', dataset.rpix()),
            rval=info.get('rval', dataset.rval()),
            rota=info.get('rota', dataset.rota()),
            dtype=info.get('dtype', str(dataset.dtype))))
    opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
    for attr in ['size', 'step', 'cval', 'scale']:
        if attr in opts:
            opts[attr] = opts[attr][:ndim]
    if 'dtype' in opts:
        opts['dtype'] = np.dtype(opts['dtype']).type
    if 'psf' in opts:
        opts['psf'] = gbkfit.psflsf.psf_parser.load(opts['psf'])
    if 'lsf' in opts:
        opts['lsf'] = gbkfit.psflsf.lsf_parser.load(opts['lsf'])
    return opts


def dump_dmodel_common(dmodel):
    info = dict(
        type=dmodel.type(),
        size=dmodel.size(),
        step=dmodel.step(),
        cval=dmodel.cval(),
        rota=dmodel.rota(),
        scale=dmodel.scale(),
        dtype=dmodel.dtype(),
        convw=dmodel.convw())
    if hasattr(dmodel, 'psf'):
        info.update(psf=gbkfit.psflsf.psf_parser.dump(dmodel.psf()))
    if hasattr(dmodel, 'lsf'):
        info.update(lsf=gbkfit.psflsf.lsf_parser.dump(dmodel.lsf()))
    return info
