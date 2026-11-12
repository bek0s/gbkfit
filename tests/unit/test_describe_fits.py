"""
Tests for the descriptions of the HDUs of FITS files from their headers
(see describe_fits).
"""

import numpy as np
import pytest
from astropy.io import fits

from gbkfit.io import describe_fits, read_data


def test_fits_files_are_described_from_their_headers(tmp_path):
    # A cube of radio velocities per beam, with its beam; its velocity
    # field; a table (not described); and a cube of CASA whose header
    # leaves out the convention of its velocities (VELO-LSR without
    # VELREF), the beam of its unit per beam, with an axis of Stokes
    sky = dict(
        CTYPE1='RA---SIN', CRVAL1=150.0, CDELT1=-1 / 3600, CRPIX1=4,
        CTYPE2='DEC--SIN', CRVAL2=2.0, CDELT2=1 / 3600, CRPIX2=3)
    cube = fits.PrimaryHDU(np.zeros((5, 6, 7), np.float32), fits.Header(
        sky | dict(CTYPE3='VRAD', CRVAL3=1e6, CDELT3=1e4, CRPIX3=1,
                   CUNIT3='m/s', BUNIT='Jy/beam', BMAJ=2 / 3600,
                   BMIN=1 / 3600, BPA=30, RMS=0.002)))
    velocity = fits.ImageHDU(
        np.zeros((6, 7)), fits.Header(sky | dict(BUNIT='km/s')),
        name='MOM1')
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name='x', format='E', array=np.zeros(3))])
    casa = fits.ImageHDU(np.zeros((5, 1, 6, 7)), fits.Header(sky | dict(
        CTYPE3='STOKES', CRVAL3=1, CDELT3=1, CRPIX3=1,
        CTYPE4='VELO-LSR', CRVAL4=1e6, CDELT4=1e4, CRPIX4=1,
        CUNIT4='m/s', BUNIT='Jy/beam')), name='CASA')
    fits.HDUList([cube, velocity, table, casa]).writeto(tmp_path / 'f.fits')
    hdus = describe_fits(str(tmp_path / 'f.fits'))
    assert [(h.index, h.name) for h in hdus] == [
        (0, 'PRIMARY'), (1, 'MOM1'), (3, 'CASA')]
    first, second, third = hdus
    assert first.shape == (7, 6, 5) and first.axes[2] == 'VRAD'
    assert (first.unit, first.rms, first.problems) == ('Jy/beam', 0.002, ())
    np.testing.assert_allclose(first.beam[:3], (2, 1, 30))
    np.testing.assert_allclose(first.coords.step, (1, 1, 10))
    assert first.coords.convention == 'radio'
    assert (second.shape, second.unit, second.rms, second.problems) == (
        (7, 6), 'km/s', None, ())
    assert third.shape == (7, 6, 1, 5) and third.coords is None
    assert (first.ambiguous_velocity, third.ambiguous_velocity) \
        == (False, True)
    (problem,) = third.problems
    assert 'is per beam, but the header has no beam' in problem
    # (its convention given, its velocities are read)
    third = describe_fits(str(tmp_path / 'f.fits'), velocity='radio')[2]
    assert third.coords.convention == 'radio'
    assert third.problems == (problem,)


def test_what_errors_hold_is_described_as_their_header_states_it(
        tmp_path):
    # (as ESO's science data products state it: MUSE, MaNGA; a header
    # that does not, nothing)
    data = fits.PrimaryHDU(np.zeros((3, 4, 5), np.float32))
    hdus = [data] + [
        fits.ImageHDU(np.zeros((3, 4, 5), np.float32), fits.Header(
            dict(HDUCLAS1='IMAGE', HDUCLAS2='ERROR', HDUCLAS3=kind)),
            name=kind)
        for kind in ('RMSE', 'MSE', 'INVMSE', 'OTHER')]
    hdus.append(fits.ImageHDU(np.zeros((3, 4, 5), np.float32), name='ERR'))
    fits.HDUList(hdus).writeto(tmp_path / 'f.fits')
    assert [h.error_kind for h in describe_fits(str(tmp_path / 'f.fits'))] == [
        None, 'sigma', 'variance', 'inverse_variance', None, None]


def test_an_axis_of_a_type_fits_does_not_define_is_read_as_given(
        tmp_path):
    # (as CALIFA writes its cubes: CTYPE3 = 'WAVELENGTH', which says
    # nothing of air or vacuum, or of linear or logarithmic wavelengths)
    header = fits.Header(dict(
        CTYPE1='RA---TAN', CDELT1=-1 / 3600, CTYPE2='DEC--TAN',
        CDELT2=1 / 3600, CTYPE3='WAVELENGTH', CRVAL3=6500, CDELT3=2,
        CRPIX3=1, CUNIT3='Angstrom'))
    fits.writeto(tmp_path / 'califa.fits', np.zeros((4, 3, 2)), header)
    filename = str(tmp_path / 'califa.fits')
    (hdu,) = describe_fits(filename, rest='6562.8 Angstrom')
    assert hdu.nonstandard_type == 'WAVELENGTH' and hdu.coords is None
    (problem,) = hdu.problems
    assert "'WAVELENGTH' (in Angstrom), is not one FITS defines" in problem
    (hdu,) = describe_fits(
        filename, rest='6562.8 Angstrom', spectral_type='AWAV')
    assert hdu.problems == () and hdu.coords.medium == 'air'
    np.testing.assert_allclose(
        hdu.coords.step[2], 299792.458 * 2 / 6562.8, rtol=1e-6)
    # (read as data, as describe reads them; a type not of FITS refused)
    with pytest.raises(ValueError, match='is not one FITS defines'):
        read_data(filename, rest='6562.8 Angstrom')
    _, coords = read_data(
        filename, rest='6562.8 Angstrom', spectral_type='AWAV')
    np.testing.assert_allclose(coords.step[2], hdu.coords.step[2])
    with pytest.raises(ValueError, match="must be one of"):
        read_data(filename, spectral_type='WAVELENGTH')
    # (a header of FITS types is read as it is: the type is of an axis of
    # a type FITS does not define)
    header['CTYPE3'] = 'AWAV'
    fits.writeto(tmp_path / 'fits.fits', np.zeros((4, 3, 2)), header)
    (hdu,) = describe_fits(
        str(tmp_path / 'fits.fits'), rest='6562.8 Angstrom',
        spectral_type='WAVE')
    assert hdu.nonstandard_type is None and hdu.coords.medium == 'air'


def test_a_cube_of_wavelengths_is_described_in_velocities_of_a_rest(
        tmp_path):
    header = fits.Header(dict(
        CTYPE1='RA---TAN', CDELT1=-0.2 / 3600, CTYPE2='DEC--TAN',
        CDELT2=0.2 / 3600, CTYPE3='AWAV', CRVAL3=6600, CDELT3=1.25,
        CRPIX3=1, CUNIT3='Angstrom'))
    fits.writeto(tmp_path / 'ifu.fits', np.zeros((4, 3, 2)), header)
    (hdu,) = describe_fits(str(tmp_path / 'ifu.fits'))
    assert hdu.coords is None and 'AWAV' in hdu.problems[0]
    (hdu,) = describe_fits(str(tmp_path / 'ifu.fits'), rest='6562.8 Angstrom')
    assert hdu.problems == ()
    np.testing.assert_allclose(
        hdu.coords.step, (0.2, 0.2, 299792.458 * 1.25 / 6562.8), rtol=1e-3)
