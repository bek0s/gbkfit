"""
Every observable with the options of the instrument, the foreground and
the galaxy together (oversampling, psf, lsf, primary beam, lens, a
selection of components, emission lines): the model is finite, of the
shape of its data, and not empty.
"""

import copy

import gbkfit.params
import numpy as np
import pytest

from gbkfit.instrument import instrument_parser
from gbkfit.model import Extras, galaxy_parser
from gbkfit.observation import (
    Flat, Foreground, Lens, LensDeflectionMap, Observation, ObservationGroup,
    observable_parser)


def component(name, **emission):
    """A thin disk that emits lines, with the given emission options."""
    return dict(
        type='smdisk', name=name, loose=False, tilted=False,
        rnodes=list(range(0, 14)),
        density=dict(ptraits=dict(type='exponential')),
        kinematics=dict(
            vptraits=dict(type='arctan', direction='tan'), dptraits=dict(type='uniform')),
        emission=dict(type='lines', **emission))


LINES = [dict(name='ha', rest='6562.8 Angstrom'),
         dict(name='nii', rest='6583.45 Angstrom')]

GALAXY = dict(components=[
    component('gas', lines=LINES), component('stars')])

PROPERTIES = {
    f'{name}_{key}': value
    for name in ('gas', 'stars')
    for key, value in dict(
        vsys=0, xpos=0.3, ypos=-0.6, posa=50, incl=60, pt_a=1, pt_s=3,
        vpt_rt=2, vpt_vt=150, dpt_a=20).items()} | dict(gas_nii_ratio=0.3)

APERTURES = dict(type='apertures', apertures=[
    dict(type='circle', x=1, y=0, radius=2), dict(type='field')])

OBSERVABLES = dict(
    pixel_spectra=dict(
        type='pixel_spectra', size=[24, 25, 61], step=[1, 1, 20],
        rest='6562.8 Angstrom'),
    slit_spectra=dict(
        type='slit_spectra', size=[24, 61], step=[1, 20], rota=40,
        rest='6562.8 Angstrom'),
    pixel_moments=dict(
        type='pixel_moments', size=[24, 25], spec_size=81, spec_step=10,
        spec_rest='6562.8 Angstrom', mask_cutoff=1e-4),
    pixel_moments_fit=dict(
        type='pixel_moments', size=[24, 25], spec_size=81, spec_step=10,
        spec_rest='6562.8 Angstrom', mask_cutoff=1e-4,
        method='gaussian_fit'),
    region_moments=dict(
        type='region_moments', regions=APERTURES, size=[24, 25],
        spec_size=81, spec_step=10, spec_rest='6562.8 Angstrom',
        mask_cutoff=1e-4),
    region_spectra=dict(
        type='region_spectra', regions=APERTURES, size=[24, 25],
        spec_size=61, spec_step=20, spec_rest='6562.8 Angstrom'))

SCALES = dict(
    pixel_spectra=[2, 2, 1], slit_spectra=[2, 1], pixel_moments=[2, 2],
    pixel_moments_fit=[2, 2], region_moments=[2, 2],
    region_spectra=[2, 2, 1])

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
    if name in ('pixel_moments', 'pixel_moments_fit', 'region_moments'):
        # Moments of a narrow spectral axis do not need the Hanning LSF
        instrument['lsf'] = dict(type='gauss', sigma=15)
    foreground = None
    if lens:
        # A constant deflection: the source moved by (1, -1) arcsec
        foreground = Foreground(Lens((40, 40), (0.5, 0.5), LensDeflectionMap(
            np.ones((40, 40)), -np.ones((40, 40)))))
    observation = Observation(
        observable_parser.load(info), Flat(), driver,
        foreground=foreground,
        instrument=instrument_parser.load(copy.deepcopy(instrument)),
        components=['gas'], scale=SCALES[name])
    group = ObservationGroup(
        [galaxy_parser.load(copy.deepcopy(GALAXY))], [observation])
    params = gbkfit.params.ParamSpace(group.pdescs(), PROPERTIES)
    extra = Extras()
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
    assert extra.find(observation='observation0', galaxy='galaxy0')
