

import numpy as np

import gbkfit.psflsf
from gbkfit.utils import parseutils


__all__ = [
    'load_dmodel_common'
]


def load_dmodel_common(
        cls, info, ndim, has_psf, has_lsf, dataset, expected_dataset_cls):
    desc = parseutils.make_typed_desc(cls, 'dmodel')
    # Load psf/lsf
    if has_psf:
        parseutils.load_option_and_update_info(
            gbkfit.psflsf.psf_parser, info, 'psf')
    if has_lsf:
        parseutils.load_option_and_update_info(
            gbkfit.psflsf.lsf_parser, info, 'lsf')
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
            dtype=info.get('dtype', np.dtype(dataset.dtype()).name)))
    parseutils.sanitize_dimensional_options(info, dict(
        size=int, step=int | float, rpix=int | float, rval=int | float,
        scale=int), ndim)
    # Parse options and create object
    opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
    return opts

