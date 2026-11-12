"""
Tests for the grids of pixels with world coordinates.
"""

import astropy.io.fits as fits
import astropy.units as u
import astropy.wcs
import numpy as np
import pytest

from gbkfit.grid import (
    CONVENTIONS, Grid, VelocityConvention, check_overlap, make_convention,
    make_grid, make_rest, pixels_on, x_axis_pa, x_axis_rota)
from gbkfit.io import coords_from_header, read_data, write_data


def celestial_wcs(crval, crpix, rota):
    """A celestial WCS of 1 arcsec pixels, east to the left, rotated."""
    wcs = astropy.wcs.WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crval = crval
    wcs.wcs.crpix = crpix
    wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]
    angle = np.radians(rota)
    wcs.wcs.pc = [
        [np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    return wcs


def test_grids_are_placed_by_their_sky_positions():
    # Two grids of different reference positions and rotations: the pixel
    # on the second of each pixel of the first is that of astropy
    wcs_a = celestial_wcs([150.0, 50.0], [20, 15], 30)
    wcs_b = celestial_wcs([150.004, 50.002], [5, 8], -10)
    grid_a = Grid(
        (40, 30), coords_from_header('a', wcs_a.to_header()), None)
    grid_b = Grid(
        (25, 20), coords_from_header('b', wcs_b.to_header()), None)
    j, i = np.mgrid[0:30, 0:40]
    expected = wcs_b.world_to_pixel(wcs_a.pixel_to_world(i, j))
    np.testing.assert_allclose(
        pixels_on(grid_a, grid_b), expected, atol=1e-6)


@pytest.mark.parametrize('position_angle', [-90, 0, 30, 135])
def test_the_x_axis_of_a_grid_is_90_degrees_from_its_rotation(
        position_angle):
    # The rotation of a grid (rota) is the position angle of its +y axis;
    # its +x axis (e.g. along a slit, whose position angle is quoted) is
    # 90 degrees clockwise of it, on the sky of north up and east to the
    # left: astropy finds the position angle of +x of the WCS of the
    # rotation of x_axis_rota
    rota = x_axis_rota(position_angle)
    wcs = celestial_wcs([150.0, 50.0], [1, 1], rota)
    coords = coords_from_header('a', wcs.to_header())
    assert (coords.rota - rota + 180) % 360 - 180 == pytest.approx(
        0, abs=1e-6)
    centre, ahead = wcs.pixel_to_world([0, 1], [0, 0])
    assert (centre.position_angle(ahead).deg - position_angle + 180) \
        % 360 - 180 == pytest.approx(0, abs=1e-6)
    assert x_axis_pa(rota) == position_angle


def test_grids_must_overlap():
    grid = make_grid((10, 10), rval=(150.0, 2.0))
    check_overlap(grid, grid, 'the image')
    far = make_grid((10, 10), rval=(150.0, 2.1))
    with pytest.raises(ValueError, match="the image does not cover"):
        check_overlap(grid, far, 'the image')


@pytest.mark.parametrize('options, message', [
    (dict(size=(10, 0)), "positive integers"),
    (dict(size=(10, 2.5)), "positive integers"),
    (dict(size=(10, 10), step=(1, 1, 1)), "option 'step' must have one"),
    (dict(size=(10, 10), step=-1), "step must be positive"),
    (dict(size=(10, 10, 5), spectral_axis=3), "spectral axis cannot be 3"),
    (dict(size=(10, 10), rest='1 GHz'), "has no rest")])
def test_make_grid_errors(options, message):
    with pytest.raises(ValueError, match=message):
        make_grid(**options)


def test_velocity_conventions_follow_the_rest():
    # Of a rest wavelength, optical by default or log; of a frequency,
    # radio; without a rest, any, none by default
    wave = make_rest('6563 Angstrom')
    freq = make_rest('1420 MHz')
    assert make_convention(None, wave) == 'optical'
    assert make_convention('log', wave) == 'log'
    assert make_convention(None, freq) == 'radio'
    assert make_convention(None, None) is None
    for convention in CONVENTIONS:
        assert make_convention(convention, None) == convention
    for convention, rest in (
            ('radio', wave), ('log', freq), ('fast', None)):
        with pytest.raises(ValueError, match="convention"):
            make_convention(convention, rest)


@pytest.mark.parametrize('rest, convention', [
    ('6563 Angstrom', None), ('6563 Angstrom', 'log'), ('1420 MHz', None),
    (None, None), (None, 'optical'), (None, 'log'), (None, 'radio')])
def test_velocity_conventions_place_light(rest, convention):
    # Each convention turns velocities into the 1 + z of their light and
    # back, and a km/s of motion (which changes 1 + z by D / c) moves the
    # light by its stretch in velocity of the axis
    c = 299792.458
    rest = None if rest is None else make_rest(rest)
    velocities = VelocityConvention(rest, None, convention)
    v = np.array([-3000.0, 0.0, 12000.0])
    d = velocities.factor(v)
    np.testing.assert_allclose(velocities.velocity(d), v, atol=1e-8)
    moved = (velocities.velocity(d * (1 + 1e-3 / c)) - v) / 1e-3
    np.testing.assert_allclose(velocities.stretch(v), moved, rtol=1e-6)


def test_log_velocities_are_of_logarithmic_wavelengths():
    # On an axis uniform in ln w, the light of the velocity v is at the
    # wavelength rest exp(v / c), in its frequencies and in the equivalency
    # of the points of tables (e.g. of a varying LSF) too
    c = 299792.458
    rest = make_rest('6564.61 Angstrom')
    velocities = VelocityConvention(rest, None, 'log')
    wavelength = 6564.61e-10 * np.exp(3000 / c)
    assert velocities.wavelength(3000) == pytest.approx(wavelength, rel=1e-12)
    assert velocities.frequency(3000) == pytest.approx(
        c * 1e3 / wavelength, rel=1e-12)
    assert velocities.velocity_at(c * 1e3 / wavelength) == pytest.approx(3000)
    assert (3000 * u.km / u.s).to_value(
        u.m, velocities.equivalency()) == pytest.approx(wavelength, rel=1e-12)


def test_logarithmic_wavelengths_are_log_velocities(tmp_path):
    # An axis of WAVE-LOG, w = S exp(x / S), is linear in c ln(w / rest):
    # of step c CDELT / S and value c ln(S / rest) at the reference pixel;
    # written back as it was
    c = 299792.458
    rest, crval, cdelt = 6564.61e-10, 6600e-10, 0.8e-10 * 6600 / 6564.61
    header = fits.Header(dict(
        CTYPE1='RA---TAN', CRVAL1=150, CRPIX1=3, CDELT1=-1 / 3600,
        CTYPE2='DEC--TAN', CRVAL2=2, CRPIX2=3, CDELT2=1 / 3600,
        CTYPE3='WAVE-LOG', CUNIT3='m', CRVAL3=crval, CRPIX3=5,
        CDELT3=cdelt, RESTWAV=rest))
    fits.writeto(tmp_path / 'cube.fits', np.zeros((9, 5, 5)), header)
    data, coords = read_data(str(tmp_path / 'cube.fits'))
    assert coords.convention == 'log'
    assert coords.step[2] == pytest.approx(c * cdelt / crval, rel=1e-12)
    assert coords.rpix[2] == pytest.approx(4)
    assert coords.rval[2] == pytest.approx(c * np.log(crval / rest),
                                           rel=1e-12)
    write_data(str(tmp_path / 'out.fits'), data, coords, 2)
    out = fits.getheader(tmp_path / 'out.fits')
    assert out['CTYPE3'] == 'WAVE-LOG'
    # (in air, AWAV-LOG, as FITS has it)
    write_data(
        str(tmp_path / 'air.fits'), data, coords._replace(medium='air'), 2)
    assert fits.getheader(tmp_path / 'air.fits')['CTYPE3'] == 'AWAV-LOG'
    _, air = read_data(str(tmp_path / 'air.fits'))
    assert (air.convention, air.medium) == ('log', 'air')
    assert air.rval[2] == pytest.approx(coords.rval[2], rel=1e-12)
    assert out['CRVAL3'] == pytest.approx(crval, rel=1e-12)
    assert out['CDELT3'] == pytest.approx(cdelt, rel=1e-12)
