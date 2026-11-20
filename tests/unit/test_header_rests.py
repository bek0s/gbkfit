"""
Tests for the rests of spectral axes that their headers state (RESTFRQ,
RESTWAV), and for axes of optical velocities linear in frequency
(FELO-HEL of AIPS, VOPT-F2W of FITS), read as radio velocities.
"""

import os

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits

from gbkfit.grid import air_refractive_index
from gbkfit.io import coords_from_header, model_header, read_data


C = 299792.458
HI = 1420405752.0

SKY = dict(
    CTYPE1='RA---TAN', CRVAL1=150.0, CRPIX1=12.5, CDELT1=-10 / 3600,
    CUNIT1='deg', CTYPE2='DEC--TAN', CRVAL2=2.0, CRPIX2=10.5,
    CDELT2=10 / 3600, CUNIT2='deg')

THINGS = os.path.expanduser(
    '~/work/dev/gbkfit/data/things/NGC_3198_NA_CUBE_THINGS.FITS')


def write_cube(path, header, shape=(30, 20, 24)):
    fits.writeto(path, np.zeros(shape, np.float32), fits.Header(header))
    return str(path)


def header_coords(header):
    """The coordinates of a header of a file, read from it alone."""
    header = fits.Header(header)
    shape = tuple(header[f'NAXIS{n}'] for n in range(header['NAXIS'], 0, -1))
    model, _ = model_header('cube.fits', header, shape)
    return coords_from_header('cube.fits', model)


def freq_cube(**keywords):
    return SKY | dict(
        CTYPE3='FREQ', CUNIT3='Hz', CRVAL3=1.415e9, CDELT3=-2.5e4,
        CRPIX3=1) | keywords


def test_the_rest_frequency_of_the_header_makes_velocities(tmp_path):
    # (radio velocities c (1 - f / f0), increasing: the axis is reversed)
    path = write_cube(tmp_path / 'cube.fits', freq_cube(RESTFRQ=HI))
    coords = read_data(path)[1]
    assert coords.convention == 'radio'
    assert coords.rest.to_value(u.Hz) == pytest.approx(HI)
    assert coords.step[2] == pytest.approx(C * 2.5e4 / HI)
    assert coords.rval[2] + (0 - coords.rpix[2]) * coords.step[2] \
        == pytest.approx(C * (1 - 1.415e9 / HI))


def test_the_old_keyword_of_the_rest_frequency_too(tmp_path):
    path = write_cube(tmp_path / 'cube.fits', freq_cube(RESTFREQ=HI))
    assert read_data(path)[1].rest.to_value(u.Hz) == pytest.approx(HI)


def test_a_stated_rest_wins_over_that_of_the_header(tmp_path):
    path = write_cube(tmp_path / 'cube.fits', freq_cube(RESTFRQ=HI))
    coords = read_data(path, rest='1.415 GHz')[1]
    assert coords.rest.to_value(u.GHz) == pytest.approx(1.415)
    assert coords.rval[2] == pytest.approx(0, abs=1e-6)


def test_a_rest_wavelength_of_the_header_is_in_vacuum(tmp_path):
    # An axis of wavelengths in air whose first channel is Halpha in air,
    # and whose header has Halpha in vacuum (RESTWAV, as FITS defines it)
    halpha_vacuum = 6564.61e-10
    halpha_air = halpha_vacuum / air_refractive_index(halpha_vacuum)
    path = write_cube(tmp_path / 'cube.fits', SKY | dict(
        CTYPE3='AWAV', CUNIT3='m', CRVAL3=halpha_air, CDELT3=1e-10,
        CRPIX3=1, RESTWAV=halpha_vacuum))
    coords = read_data(path)[1]
    assert coords.medium == 'air'
    assert coords.rest.to_value(u.m) == pytest.approx(halpha_air, rel=1e-12)
    assert coords.rval[2] == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize('keywords', [{}, dict(RESTFRQ=0.0)])
def test_without_a_rest_anywhere_it_is_missing(tmp_path, keywords):
    path = write_cube(tmp_path / 'cube.fits', freq_cube(**keywords))
    with pytest.raises(ValueError, match="FREQ axis: rest missing") as e:
        read_data(path)
    assert e.value.config_path == ['spectral', 'rest']


def felo_of(header):
    """
    The header of the optical velocities linear in frequency (FELO-HEL)
    of the same channels as a header of a linear axis of frequencies,
    with the rest of the header.
    """
    f0 = header.get('RESTFRQ', header.get('RESTFREQ'))
    fr, df = header['CRVAL3'], header['CDELT3']
    felo = fits.Header(header).copy()
    felo.update(
        CTYPE3='FELO-HEL', CUNIT3='m/s', CRVAL3=C * 1e3 * (f0 / fr - 1),
        CDELT3=-C * 1e3 * f0 * df / fr ** 2)
    return felo


def test_felo_axes_are_the_radio_velocities_of_their_frequencies():
    header = freq_cube(RESTFRQ=HI, NAXIS=3, NAXIS1=24, NAXIS2=20, NAXIS3=30)
    expected = header_coords(header)
    coords = header_coords(felo_of(header))
    assert coords.convention == 'radio'
    assert coords.rest.to_value(u.Hz) == pytest.approx(HI)
    np.testing.assert_allclose(coords.step, expected.step, rtol=1e-12)
    np.testing.assert_allclose(coords.rpix, expected.rpix, atol=1e-9)
    np.testing.assert_allclose(coords.rval, expected.rval, rtol=1e-12)


def test_a_felo_axis_of_another_stated_rest(tmp_path):
    header = freq_cube(RESTFRQ=HI)
    path = write_cube(tmp_path / 'cube.fits', felo_of(header))
    expected = read_data(
        write_cube(tmp_path / 'freq.fits', header), rest='1.42 GHz')[1]
    coords = read_data(path, rest='1.42 GHz')[1]
    np.testing.assert_allclose(coords.step, expected.step, rtol=1e-12)
    np.testing.assert_allclose(coords.rval, expected.rval, rtol=1e-12)


def test_a_felo_axis_without_a_rest_takes_the_stated_one(tmp_path):
    header = freq_cube(RESTFRQ=HI)
    felo = felo_of(header)
    del felo['RESTFRQ']
    path = write_cube(tmp_path / 'cube.fits', felo)
    with pytest.raises(
            ValueError, match="FELO-HEL axis: rest missing$") as e:
        read_data(path)
    assert e.value.config_path == ['spectral', 'rest']
    expected = read_data(write_cube(tmp_path / 'freq.fits', header))[1]
    coords = read_data(path, rest=f'{HI} Hz')[1]
    np.testing.assert_allclose(coords.rval, expected.rval, rtol=1e-12)


@pytest.mark.skipif(not os.path.exists(THINGS), reason="no THINGS cube")
def test_the_header_of_a_things_cube():
    # A cube of frequencies with its rest (RESTFREQ), and the same channels
    # written as FELO-HEL: the same radio velocities
    header = fits.getheader(THINGS)
    coords = header_coords(header)
    assert coords.convention == 'radio'
    assert coords.rest.to_value(u.Hz) == pytest.approx(HI)
    assert coords.step[2] == pytest.approx(C * 24414.0625 / HI, rel=1e-9)
    felo = header_coords(felo_of(header))
    np.testing.assert_allclose(felo.step, coords.step, rtol=1e-12)
    np.testing.assert_allclose(felo.rval, coords.rval, rtol=1e-12)
