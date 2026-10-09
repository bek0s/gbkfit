"""
Every observable with the options of the instrument, the foreground and
the gmodel together (oversampling, psf, lsf, primary beam, lens, a
selection of components, emission lines): the model is finite, of the
shape of its data, and not empty.
"""

import copy

import gbkfit.params
import numpy as np
import pytest

from gbkfit.model import gmodel_parser
from gbkfit.observation import (
    Foreground, LensDeflectionMap, Observation, ObservationGroup,
    instrument_parser, observable_parser)


def component(name, **options):
    return dict(
        type='smdisk', name=name, loose=False, tilted=False,
        rnodes=list(range(0, 14)),
        bptraits=dict(type='exponential'),
        vptraits=dict(type='tan_arctan'),
        dptraits=dict(type='uniform')) | options


LINES = [dict(name='ha', rest='6562.8 Angstrom'),
         dict(name='nii', rest='6583.45 Angstrom')]

GMODEL = dict(type='kinematics_2d', components=[
    component('gas', lines=LINES), component('stars')])

PROPERTIES = {
    f'{name}_{key}': value
    for name in ('gas', 'stars')
    for key, value in dict(
        vsys=0, xpos=0.3, ypos=-0.6, posa=50, incl=60, bpt_a=1, bpt_s=3,
        vpt_rt=2, vpt_vt=150, dpt_a=20).items()} | dict(gas_nii_ratio=0.3)

APERTURES = dict(type='apertures', apertures=[
    dict(type='circle', x=1, y=0, radius=2), dict(type='field')])

OBSERVABLES = dict(
    scube=dict(type='scube', size=[24, 25, 61], step=[1, 1, 20],
               rest='6562.8 Angstrom'),
    lslit=dict(type='lslit', size=[24, 61], step=[1, 20], rota=40,
               rest='6562.8 Angstrom'),
    mmaps=dict(type='mmaps', size=[24, 25], spec_size=81, spec_step=10,
               spec_rest='6562.8 Angstrom', mask_cutoff=1e-4),
    mmaps_fit=dict(type='mmaps', size=[24, 25], spec_size=81, spec_step=10,
                   spec_rest='6562.8 Angstrom', mask_cutoff=1e-4,
                   method='gaussian_fit'),
    bmaps=dict(type='bmaps', regions=APERTURES, size=[24, 25],
               spec_size=81, spec_step=10, spec_rest='6562.8 Angstrom',
               mask_cutoff=1e-4),
    aspec=dict(type='aspec', regions=APERTURES, size=[24, 25],
               spec_size=61, spec_step=20, spec_rest='6562.8 Angstrom'))

SCALES = dict(scube=[2, 2, 1], lslit=[2, 1], mmaps=[2, 2], mmaps_fit=[2, 2],
              bmaps=[2, 2], aspec=[2, 2, 1])

INSTRUMENT = dict(
    primary_beam=dict(type='gauss', fwhm=30),
    psf=dict(type='sum', psfs=[dict(type='gauss', sigma=1),
                               dict(type='moffat', alpha=2, beta=3)],
             weights=[0.6, 0.4]),
    lsf=dict(type='convolution', lsfs=[dict(type='gauss', sigma=15),
                                       dict(type='hanning', width=20)]))


@pytest.mark.parametrize('lens', [False, True], ids=['plain', 'lensed'])
@pytest.mark.parametrize('name', list(OBSERVABLES))
def test_feature_combinations(driver, name, lens):
    info = copy.deepcopy(OBSERVABLES[name])
    instrument = dict(INSTRUMENT)
    if name in ('mmaps', 'mmaps_fit', 'bmaps'):
        # Moments of a narrow spectral axis do not need the Hanning LSF
        instrument['lsf'] = dict(type='gauss', sigma=15)
    foreground = None
    if lens:
        # A constant deflection: the source moved by (1, -1) arcsec
        foreground = Foreground(LensDeflectionMap(
            np.ones((40, 40)), -np.ones((40, 40)), (40, 40), (0.5, 0.5)))
    observation = Observation(
        driver, observable_parser.load(info),
        foreground=foreground,
        instrument=instrument_parser.load(copy.deepcopy(instrument)),
        components=['gas'], scale=SCALES[name])
    group = ObservationGroup(
        [gmodel_parser.load(copy.deepcopy(GMODEL))], [observation])
    params = gbkfit.params.EvaluationParams(group.pdescs(), PROPERTIES)
    extra = {}
    data = group.model_h(params.evaluate(), extra)[0]
    observable = observation.observable()
    assert set(data) == set(observable.keys())
    for key, value in data.items():
        model = value['d']
        assert np.isfinite(model).any(), key
        assert np.nanmax(np.abs(model)) > 0, key
        if value['m'] is None:
            assert np.isfinite(model).all(), key
    # The extra outputs can be written as their observables give them
    for key, value in data.items():
        observable.output(value['d'])
    assert any(key.startswith('observation0_gmodel_') for key in extra)
