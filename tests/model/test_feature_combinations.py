"""
Every kind of observation with the options of the instrument, a lens
and the galaxy together (oversampling, psf, lsf, primary beam, lens, a
selection of components, emission lines): the model is finite, of the
shape of its data, and not empty.
"""

import copy

import gbkfit.params
import numpy as np
import pytest

from gbkfit.instrument import instrument_parser
from gbkfit.lensing import Lens, LensDeflectionMap
from gbkfit.model import Extras, galaxy_parser
from gbkfit.observation import ObservationGroup, observation_parser


def component(name, **emission):
    """A thin disk that emits lines, with the given emission options."""
    return dict(
        type='thin_disk', name=name, loose=False, tilted=False,
        rnodes=list(range(0, 14)),
        rptraits=dict(type='exponential'),
        vptraits=dict(type='arctan', direction='tan'),
        dptraits=dict(type='uniform'),
        emission=dict(type='lines', **emission))


LINES = [dict(name='ha', rest='6562.8 Angstrom'),
         dict(name='nii', rest='6583.45 Angstrom')]

GALAXY = dict(components=[
    component('gas', lines=LINES), component('stars')])

# The redshift of the galaxy behind a lens, and of the lens: the
# spectral axes of a lensed galaxy are of its lines at its redshift
SOURCE_REDSHIFT = 1.0
LENS_REDSHIFT = 0.3

# The geometry of the gas, which that of the stars is tied to
GEOMETRY = dict(vsys=0, xpos=0.3, ypos=-0.6, posa=50, incl=60)
PROPERTIES = {f'gas_{key}': value for key, value in GEOMETRY.items()} | {
    f'stars_{key}': f'gas_{key}' for key in GEOMETRY} | {
    f'{name}_{key}': value
    for name in ('gas', 'stars')
    for key, value in dict(
        rpt_a=1, rpt_s=3, vpt_rt=2, vpt_vt=150, dpt_a=20).items()} \
    | dict(gas_nii_ratio=0.3)

APERTURES = dict(type='apertures', apertures=[
    dict(type='circle', x=1, y=0, radius=2), dict(type='field')])

KINDS = dict(
    pixel_spectra=dict(
        type='pixel_spectra', size=[24, 25], step=[1, 1],
        spectral=dict(size=61, step=20, rest='6562.8 Angstrom')),
    slit_spectra=dict(
        type='slit_spectra', size=[24], step=[1], slit_pa=-50,
        spectral=dict(size=61, step=20, rest='6562.8 Angstrom')),
    pixel_moments=dict(
        type='pixel_moments', size=[24, 25],
        spectral=dict(size=81, step=10, rest='6562.8 Angstrom'),
        flux_floor=1e-4),
    pixel_moments_fit=dict(
        type='pixel_moments', size=[24, 25],
        spectral=dict(size=81, step=10, rest='6562.8 Angstrom'),
        flux_floor=1e-4, method='gauss_fit'),
    region_moments=dict(
        type='region_moments', regions=APERTURES, size=[24, 25],
        spectral=dict(size=81, step=10, rest='6562.8 Angstrom'),
        flux_floor=1e-4),
    region_spectra=dict(
        type='region_spectra', regions=APERTURES, size=[24, 25],
        spectral=dict(size=61, step=20, rest='6562.8 Angstrom')))

OVERSAMPLING = dict(
    pixel_spectra=[2, 2, 1], slit_spectra=[2, 1], pixel_moments=[2, 2],
    pixel_moments_fit=[2, 2], region_moments=[2, 2],
    region_spectra=[2, 2, 1])

INSTRUMENT = dict(
    primary_beam=dict(type='gauss', fwhm=30),
    psf=dict(type='sum', psfs=[dict(type='gauss', sigma=1),
                               dict(type='moffat', alpha=2, beta=3)],
             weights=[0.6, 0.4]),
    lsf=dict(type='convolution', lsfs=[
        dict(type='gauss', sigma='15 km/s'),
        dict(type='hanning')]))


@pytest.mark.parametrize('lens', [False, True], ids=['plain', 'lensed'])
@pytest.mark.parametrize('name', list(KINDS))
def test_feature_combinations(driver, name, lens):
    info = copy.deepcopy(KINDS[name])
    instrument = dict(INSTRUMENT)
    moments = name in ('pixel_moments', 'pixel_moments_fit', 'region_moments')
    if moments:
        # Moments of a narrow spectral axis do not need the Hanning LSF
        instrument['lsf'] = dict(
            type='gauss', sigma='15 km/s')
    galaxy = copy.deepcopy(GALAXY)
    lenses = []
    if lens:
        # A constant deflection: the source moved by (1, -1) arcsec
        lenses = [Lens('lens', LENS_REDSHIFT, LensDeflectionMap(
            np.ones((80, 80)), -np.ones((80, 80))))]
        galaxy['redshift'] = SOURCE_REDSHIFT
        info['spectral']['rest'] = \
            f'{6562.8 * (1 + SOURCE_REDSHIFT)} Angstrom'
    observation = observation_parser.load(info | dict(
        instrument=instrument_parser.load(copy.deepcopy(instrument)),
        oversampling=OVERSAMPLING[name],
        # Moments are of one line (they would mix the lines of a spectrum)
        lines=['ha'] if moments else None))
    group = ObservationGroup(
        [galaxy_parser.load(galaxy)], [observation], lenses, seen=[['gas']])
    params = gbkfit.params.ParamSpace.from_properties(
        group.pdescs(), PROPERTIES)
    extra = Extras()
    data = group.plan(driver, reference=params.evaluate()).model_h(
        params.evaluate(), extra)[0]
    assert set(data) == set(observation.keys())
    for key, value in data.items():
        model = value.data
        assert np.isfinite(model).any(), key
        assert np.nanmax(np.abs(model)) > 0, key
        if value.mask is None:
            assert np.isfinite(model).all(), key
    # The extra outputs can be written as their observations give them
    for key, value in data.items():
        observation.output(value.data)
    assert extra.find(observation='observation0', galaxy='galaxy0')
