"""
Tests for the bands: components of a continuum in bands, the band of the
tracer of an observation, and dust that dims an image at the rest
wavelength of its band. GBKFIT knows no bands: the user declares them.
"""

import numpy as np
import pytest
from modelutils import config_group_plan, group_plan

from gbkfit.model import LineCatalogue, lines_from


# The bands that the tests declare (those of SDSS g, r and i)
BANDS = {'g': '4686 Angstrom', 'r': '6166 Angstrom', 'i': '7480 Angstrom'}


@pytest.fixture(autouse=True)
def declared_bands():
    """The bands of the tests, declared for the components they make."""
    with lines_from(LineCatalogue(bands=BANDS)):
        yield


DISK = dict(loose=False, tilted=False, rnodes=list(range(0, 12)))

STARS = dict(
    DISK, type='thin_disk', name='stars',
    rptraits=dict(type='exponential'),
    light=dict(type='continuum', bands=['g', 'r']))

NUCLEUS = dict(
    type='point', name='nucleus',
    light=dict(type='continuum', bands=['g', 'r']))

PROPERTIES = dict(
    stars=dict(
        stars_xpos=0, stars_ypos=0, stars_posa=0, stars_incl=0,
        stars_rpt_a=1, stars_rpt_s=4, stars_r_ratio=0.5),
    nucleus=dict(
        nucleus_xpos=0.5, nucleus_ypos=0.5, nucleus_flux=1,
        nucleus_r_ratio=0.5))


def image_case(driver, components, band, depth=None, redshift=0):
    """
    An image in a band of a galaxy of the given components, on the given
    depth axis, if it needs one.
    """
    return dict(
        driver=dict(type=driver.type()),
        observation=dict(type='pixel_brightness', size=[24, 24]),
        galaxy=dict(components=components, redshift=redshift),
        **({} if band is None
           else dict(tracer=dict(type='continuum', band=band))),
        **({} if depth is None else dict(depth=depth)))


def test_the_catalogue_has_the_bands_of_the_user_only():
    catalogue = LineCatalogue(bands={'mine': '5000 Angstrom'})
    assert catalogue.band('mine').wavelength().to_value('um') \
        == pytest.approx(0.5)
    # (GBKFIT knows no bands of surveys)
    with pytest.raises(ValueError, match="there is no band 'sdss_r'"):
        catalogue.band('sdss_r')


@pytest.mark.parametrize('emitter', ['stars', 'nucleus'])
def test_light_in_each_band(driver, evaluate_cases, emitter):
    # The light in the first band is the density; that in another band is
    # times its flux ratio
    component = dict(stars=STARS, nucleus=NUCLEUS)[emitter]
    images = [
        evaluate_cases(
            [image_case(driver, [component], band)],
            PROPERTIES[emitter])[0][0]['brightness'].data
        for band in ('g', 'r')]
    assert images[0].sum() > 0
    np.testing.assert_allclose(images[1], 0.5 * images[0], rtol=1e-6)


@pytest.mark.parametrize('band, message', [
    (None, "the tracer of the observation has no band"),
    ('i', "it sees none")])
def test_images_see_components_in_their_band(driver, band, message):
    with pytest.raises(ValueError, match=message):
        group_plan([image_case(driver, [STARS], band)])


def test_a_spectral_observation_has_no_tracer_of_a_continuum(driver):
    case = dict(
        driver=dict(type=driver.type()),
        observation=dict(
            type='pixel_spectra', size=[8, 8],
            spectral=dict(size=10, convention='relativistic')),
        galaxy=dict(components=[dict(
            STARS, light=dict(type='lines'),
            vptraits=dict(type='arctan', direction='tan'),
            dptraits=dict(type='uniform'))]),
        tracer=dict(type='continuum', band='r'))
    with pytest.raises(ValueError, match="takes the tracers"):
        group_plan([case])


@pytest.mark.parametrize('redshift', [0, 0.5])
def test_dust_dims_an_image_at_the_rest_wavelength_of_its_band(
        driver, evaluate_cases, redshift):
    # A face-on thin disk in the middle of face-on dust of a law whose
    # ratio is 3 - 2.5 (wavelength - 0.2): its light crosses half of the
    # optical depth at the rest wavelength of the band (0.6166 / (1 + z))
    dust = dict(
        DISK, type='thick_disk', name='dust',
        rptraits=dict(type='exponential'),
        rhtraits=dict(type='sech2', zmax=8),
        opacity=dict(type='dust', law=dict(
            type='table', wavelengths=[0.2, 1.0], ratios=[3.0, 1.0])))
    depth = dict(extent=40 * 0.25, step=0.25)
    properties = PROPERTIES['stars'] | dict(
        dust_xpos=0, dust_ypos=0, dust_posa=0, dust_incl=0, dust_rpt_s=4,
        dust_rht_s=1)

    def evaluate(opacity):
        case = image_case(
            driver, [STARS, dust], 'r', depth, redshift)
        data, extra = evaluate_cases(
            [case], properties | dict(dust_rpt_a=opacity))
        tau = extra.get('odata', component='dust').data.sum(axis=0)
        return data[0]['brightness'].data, tau
    clear, _ = evaluate(0)
    dimmed, tau = evaluate(0.5)
    ratio = 3 - 2.5 * (0.6166 / (1 + redshift) - 0.2)
    visible = clear > 1e-3 * clear.max()
    np.testing.assert_allclose(
        dimmed[visible] / clear[visible], np.exp(-ratio * tau[visible] / 2),
        rtol=1e-4)


def test_configurations_name_bands():
    config = dict(
        bands=dict(g='4686 Angstrom', mine='5000 Angstrom'),
        galaxies=[dict(components=[dict(
            STARS, light=dict(type='continuum', bands=['g', 'mine']))])],
        observations=[dict(
            type='pixel_brightness',
            size=[8, 8],
            tracer=dict(type='continuum', band='mine'))])
    plan = config_group_plan(config)
    observation, = plan.group().observations()
    assert observation.tracer().band().name() == 'mine'
    component, = plan.group().galaxies()[0].components()
    assert component.light().bands().names() == ('g', 'mine')
    assert 'stars_mine_ratio' in plan.group().pdescs()
