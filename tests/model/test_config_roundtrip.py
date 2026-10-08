"""
Configurations survive being dumped and loaded again: loading one,
dumping it, loading the dump and dumping that gives the same dump (which
is plain data, so it can be written as YAML or JSON), and the same model.
"""

import copy
import json
import pathlib

import numpy as np
import pytest
import ruamel.yaml

import gbkfit.driver
import gbkfit.model
import gbkfit.params
import gbkfit.psflsf
from gbkfit.model.gmodels import common, traits
from gbkfit.utils import funcutils


DATA_DIR = pathlib.Path(__file__).parents[1] / 'data'

CONFIGS = sorted(DATA_DIR.glob('*/*.yaml'))

# The types with a custom dump, which writes data files
FILE_TYPES = ('image', 'array')

# Values for the required options of the types tested below
SAMPLE_OPTIONS = dict(
    nblobs=2, order=1, origin=0, alpha=1.5, beta=2, sigma=1, gamma=1,
    axis=0, angle=10, weight=2)


def roundtrip(parser, info):
    """Load, dump, load and dump; return the second dump."""
    dumped = parser.dump(parser.load(copy.deepcopy(info)))
    json.dumps(dumped)
    again = parser.dump(parser.load(copy.deepcopy(dumped)))
    assert again == dumped
    return again


def sample_info(cls):
    """A configuration of a type, with its required options."""
    required = funcutils.extract_args(cls.__init__)[1]
    return dict(type=cls.type()) | {
        name: SAMPLE_OPTIONS[name] for name in required}


# The types whose dump loses options
LOSSY_DUMPS = ('axis_range',)


def registered_types(parsers):
    return [
        pytest.param(
            parser, cls, id=f'{name}-{cls.type()}',
            marks=[pytest.mark.xfail(
                strict=True, reason="the dump loses options")]
            if cls.type() in LOSSY_DUMPS else [])
        for name, parser in parsers.items()
        for cls in parser._parsers.values()
        if cls.type() not in FILE_TYPES]


TRAIT_PARSERS = {
    name: getattr(traits, name)
    for name in dir(traits) if name.endswith('_parser')}

OTHER_PARSERS = dict(
    psf=gbkfit.psflsf.psf_parser,
    lsf=gbkfit.psflsf.lsf_parser,
    nwmode=common.nwmode_parser,
    driver=gbkfit.driver.driver_parser)


@pytest.mark.parametrize(
    'parser, cls', registered_types(TRAIT_PARSERS | OTHER_PARSERS))
def test_type_roundtrip(parser, cls):
    roundtrip(parser, sample_info(cls))


def test_pdescs_roundtrip():
    info = dict(
        a=dict(type='scalar', minimum=0, maximum=1, desc='a scalar'),
        v=dict(type='vector', size=3, default=2))
    pdescs = gbkfit.params.load_pdescs_dict(copy.deepcopy(info))
    dumped = gbkfit.params.dump_pdescs_dict(pdescs)
    again = gbkfit.params.dump_pdescs_dict(
        gbkfit.params.load_pdescs_dict(copy.deepcopy(dumped)))
    assert again == dumped


@pytest.mark.xfail(
    strict=True, reason="dmodel dumps call cval(), which no longer exists")
@pytest.mark.parametrize(
    'config', CONFIGS, ids=[f'{c.parent.name}/{c.stem}' for c in CONFIGS])
def test_model_roundtrip(config, evaluate_models):
    info = ruamel.yaml.YAML(typ='safe').load(config)
    models = info['models']
    dumped = roundtrip(gbkfit.model.model_parser, models)
    # Both the configuration and its dump give the same model
    properties = info['params']['properties']
    data, _ = evaluate_models(models, properties)
    data_dumped, _ = evaluate_models(dumped, properties)
    for model, model_dumped in zip(data, data_dumped):
        for key, value in model.items():
            np.testing.assert_allclose(
                model_dumped[key]['d'], value['d'],
                rtol=1e-5, atol=1e-6 * np.abs(value['d']).max())
