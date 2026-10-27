"""
Tests for the bands of images: components that emit in bands, the band
of an image, and dust that dims an image at the rest wavelength of its
band.
"""

import numpy as np
import pytest
from modelutils import config_group_plan, group_plan

from gbkfit.model import LineCatalogue
from gbkfit.utils.parseutils import ConfigError


DISK = dict(
    type='disk', loose=False, tilted=False, rnodes=list(range(0, 12)))

STARS = dict(
    DISK, name='stars', density=dict(ptraits=dict(type='exponential')),
    emission=dict(type='band', bands=['sdss_g', 'sdss_r']))

NUCLEUS = dict(
    type='point', name='nucleus',
    emission=dict(type='band', bands=['sdss_g', 'sdss_r']))

PROPERTIES = dict(
    stars=dict(
        stars_xpos=0, stars_ypos=0, stars_posa=0, stars_incl=0,
        stars_pt_a=1, stars_pt_s=4, stars_sdss_r_ratio=0.5),
    nucleus=dict(
        nucleus_xpos=0.5, nucleus_ypos=0.5, nucleus_flux=1,
        nucleus_sdss_r_ratio=0.5))


def image_case(driver, components, band, line_of_sight=None, redshift=0):
    """An image in a band of a galaxy of the given components."""
    return dict(
        driver=dict(type=driver.type()),
        observable=dict(type='pixel_brightness', size=[24, 24]),
        line_of_sight=line_of_sight or dict(type='flat'),
        galaxy=dict(components=components, redshift=redshift),
        **({} if band is None else dict(band=band)))


def test_the_catalogue_has_bands_and_those_of_the_user():
    catalogue = LineCatalogue(bands={'mine': '5000 Angstrom'})
    assert catalogue.band('sdss_r').wavelength().to_value('um') \
        == pytest.approx(0.6166)
    assert catalogue.band('twomass_ks').wavelength().to_value('um') \
        == pytest.approx(2.159)
    assert catalogue.band('mine').wavelength().to_value('um') \
        == pytest.approx(0.5)
    with pytest.raises(ConfigError, match="there is no band 'sdss_y'"):
        catalogue.band('sdss_y')


@pytest.mark.parametrize('emitter', ['stars', 'nucleus'])
def test_light_in_each_band(driver, evaluate_cases, emitter):
    # The light in the first band is the density; that in another band is
    # times its flux ratio
    component = dict(stars=STARS, nucleus=NUCLEUS)[emitter]
    images = [
        evaluate_cases(
            [image_case(driver, [component], band)],
            PROPERTIES[emitter])[0][0]['brightness']['d']
        for band in ('sdss_g', 'sdss_r')]
    assert images[0].sum() > 0
    np.testing.assert_allclose(images[1], 0.5 * images[0], rtol=1e-6)


@pytest.mark.parametrize('band, message', [
    (None, "the image that sees it needs a band"),
    ('sdss_i', "it sees none")])
def test_images_see_components_in_their_band(driver, band, message):
    with pytest.raises(ConfigError, match=message):
        group_plan([image_case(driver, [STARS], band)])


def test_only_images_have_a_band(driver):
    case = dict(
        driver=dict(type=driver.type()),
        observable=dict(type='pixel_spectra', size=[8, 8, 10]),
        line_of_sight=dict(type='flat'),
        galaxy=dict(components=[dict(
            STARS, emission=dict(type='lines'),
            kinematics=dict(
                type='traits', vptraits=dict(type='arctan', direction='tan'),
                dptraits=dict(type='uniform')))]),
        band='sdss_r')
    with pytest.raises(ConfigError, match="only images have a band"):
        group_plan([case])


@pytest.mark.parametrize('redshift', [0, 0.5])
def test_dust_dims_an_image_at_the_rest_wavelength_of_its_band(
        driver, evaluate_cases, redshift):
    # A face-on thin disk in the middle of face-on dust of a law whose
    # ratio is 3 - 2.5 (wavelength - 0.2): its light crosses half of the
    # optical depth at the rest wavelength of the band (0.6166 / (1 + z))
    dust = dict(
        DISK, name='dust',
        density=dict(
            ptraits=dict(type='exponential'),
            htraits=dict(type='sech2', zmax=8)),
        opacity=dict(type='dust', law=dict(
            type='table', wavelengths=[0.2, 1.0], ratios=[3.0, 1.0])))
    line_of_sight = dict(
        type='grid', depth=dict(extent=40 * 0.25, step=0.25))
    properties = PROPERTIES['stars'] | dict(
        dust_xpos=0, dust_ypos=0, dust_posa=0, dust_incl=0, dust_pt_s=4,
        dust_ht_s=1)

    def evaluate(opacity):
        case = image_case(
            driver, [STARS, dust], 'sdss_r', line_of_sight, redshift)
        data, extra = evaluate_cases(
            [case], properties | dict(dust_pt_a=opacity))
        tau = extra.get('total_odata').data.sum(axis=0)
        return data[0]['brightness']['d'], tau
    clear, _ = evaluate(0)
    dimmed, tau = evaluate(0.5)
    ratio = 3 - 2.5 * (0.6166 / (1 + redshift) - 0.2)
    visible = clear > 1e-3 * clear.max()
    np.testing.assert_allclose(
        dimmed[visible] / clear[visible], np.exp(-ratio * tau[visible] / 2),
        rtol=1e-4)


def test_configurations_name_bands():
    config = dict(
        bands=dict(mine='5000 Angstrom'),
        galaxies=[dict(components=[dict(
            STARS, emission=dict(type='band', bands=['sdss_g', 'mine']))])],
        observations=[dict(
            type='sky',
            observable=dict(type='pixel_brightness', size=[8, 8]),
            line_of_sight=dict(type='flat'), band='mine')])
    plan = config_group_plan(config)
    observation, = plan.group().observations()
    assert observation.band().name() == 'mine'
    component, = plan.group().galaxies()[0].components()
    assert component.emission().bands().names() == ('sdss_g', 'mine')
    assert 'stars_mine_ratio' in plan.group().pdescs()
