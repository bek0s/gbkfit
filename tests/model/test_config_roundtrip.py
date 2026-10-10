"""
Configurations survive being dumped and loaded again: loading one,
dumping it, loading the dump and dumping that gives the same dump (which
is plain data, so it can be written as YAML or JSON), and the same model.
"""

import copy
import io
import json
import pathlib

import numpy as np
import pytest
import ruamel.yaml

import gbkfit.dataset
import gbkfit.region
import gbkfit.driver
import gbkfit.model
import gbkfit.observation
import gbkfit.params
import gbkfit.instrument
from gbkfit.model.components.disks import traits
from gbkfit.utils import funcutils
from modelutils import config_group


DATA_DIR = pathlib.Path(__file__).parents[1] / 'data'

CONFIGS = sorted(DATA_DIR.glob('*/*.yaml'))

# The types with a custom dump, which writes data files
FILE_TYPES = ('image', 'images', 'array')

# Values for the required options of the types tested below
SAMPLE_OPTIONS = dict(
    nblobs=2, order=1, origin=0, alpha=1.5, beta=2, sigma=1, gamma=1,
    axis=0, angle=10, weight=2,
    psfs=[dict(type='gauss', sigma=1), dict(type='point')],
    lsfs=[dict(type='gauss', sigma=1), dict(type='point')],
    weights=[1, 2], bmaj=3, bmin=2,
    fwhm=10, x=1, y=2, radius=3, a=2, b=1, length=4, width=1,
    vertices=[[0, 0], [1, 0], [0, 1]])


def roundtrip(parser, info):
    """Load, dump, load and dump; return the second dump."""
    dumped = parser.dump(parser.load(copy.deepcopy(info)))
    json.dumps(dumped)
    ruamel.yaml.YAML(typ='safe').dump(dumped, io.StringIO())
    again = parser.dump(parser.load(copy.deepcopy(dumped)))
    assert again == dumped
    return again


def sample_info(cls):
    """A configuration of a type, with its required options."""
    required = funcutils.parameter_names(cls.__init__).required
    return dict(type=cls.type()) | {
        name: SAMPLE_OPTIONS[name] for name in required}


def registered_types(parsers):
    return [
        pytest.param(parser, cls, id=f'{name}-{cls.type()}')
        for name, parser in parsers.items()
        for cls in parser.registered_classes().values()
        if cls.type() not in FILE_TYPES]


TRAIT_PARSERS = {
    name: getattr(traits, name)
    for name in dir(traits) if name.endswith('_parser')}

OTHER_PARSERS = dict(
    psf=gbkfit.instrument.psf_parser,
    lsf=gbkfit.instrument.lsf_parser,
    aperture=gbkfit.region.aperture_parser,
    primary_beam=gbkfit.instrument.primary_beam_parser,
    param_mode=gbkfit.params.param_mode_parser,
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


@pytest.mark.parametrize(
    'config', CONFIGS, ids=[f'{c.parent.name}/{c.stem}' for c in CONFIGS])
def test_config_roundtrip(config):
    from gbkfit.model import model_parser
    from gbkfit.observation import observation_parser
    info = ruamel.yaml.YAML(typ='safe').load(config)
    dumped = dict(
        models=roundtrip(model_parser, info['models']),
        observations=roundtrip(observation_parser, info['observations']))
    # Both the configuration and its dump give the same model
    properties = info['params']['properties']

    def evaluate(config_):
        group = config_group(config_)
        params = gbkfit.params.EvaluationParams(
            group.pdescs(), properties, constants=group.constants())
        return group.model_h(params.evaluate())

    data = [{k: dict(v) for k, v in item.items()} for item in evaluate(info)]
    data_dumped = evaluate(dumped)
    for model, model_dumped in zip(data, data_dumped):
        for key, value in model.items():
            np.testing.assert_allclose(
                model_dumped[key]['d'], value['d'],
                rtol=1e-5, atol=1e-6 * np.nanmax(np.abs(value['d'])))


GAUSS = dict(type='gauss', sigma=1)

# Each type of observable, in an observation with every option set to a
# value other than its default
OBSERVABLES = dict(
    pixel_brightness=(dict(
        size=[20, 16], step=[2, 1], rpix=[3, 4], rval=[1, 2], rota=10,
        mask_cutoff=0.1, mask_apply=True),
        dict(psf=GAUSS, lsf=None), [2, 1]),
    pixel_spectra=(dict(
        size=[20, 16, 11], step=[2, 1, 10], rpix=[3, 4, 5], rval=[1, 2, 3],
        rota=10, smooth_weights=True, mask_cutoff=0.1, mask_apply=True),
        dict(psf=GAUSS, lsf=GAUSS), [2, 1, 1]),
    slit_spectra=(dict(
        size=[20, 11], step=[2, 10], rpix=[3, 5], rval=[1, 3], rota=10,
        slit_width=3, smooth_weights=True, mask_cutoff=0.1, mask_apply=True),
        dict(psf=GAUSS, lsf=GAUSS), [2, 1]),
    pixel_moments=(dict(
        size=[20, 16], step=[2, 1], rpix=[3, 4], rval=[1, 2], rota=10,
        mask_cutoff=0.1, orders=[0, 1], spec_size=201, spec_step=2,
        spec_rval=1500),
        dict(psf=GAUSS, lsf=GAUSS), [2, 1]))


@pytest.mark.parametrize('observable_type', OBSERVABLES)
def test_observation_dump_has_every_option(observable_type):
    from gbkfit.observation import observation_parser
    observable, instrument, scale = OBSERVABLES[observable_type]
    info = dict(
        name='obs', model='galaxy', driver=dict(type='host'),
        instrument=instrument,
        observable=dict(type=observable_type) | observable,
        scale=scale, dtype='float64')
    dumped = json.loads(json.dumps(roundtrip(observation_parser, info)))
    for key in ('name', 'model', 'scale', 'dtype'):
        assert dumped[key] == info[key], key
    for key, value in info['observable'].items():
        assert dumped['observable'][key] == value, key
    for key, value in instrument.items():
        if value is None:
            assert dumped['instrument'][key] is None
        else:
            assert dumped['instrument'][key]['sigma'] == value['sigma']
