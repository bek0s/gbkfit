"""
Tests for the descriptions of the HDUs of FITS files from their headers
(see describe_fits).
"""

import numpy as np
import pytest
from astropy.io import fits

from gbkfit.config import ConfigProblem
from gbkfit.io import describe_fits, read_data, unstated_spectral_axis


# The references of the axes on the sky (CRPIXn and CRVALn), which FITS
# gives defaults (0), and GBKFIT never takes (see reference_errors)
REFERENCES = dict(CRVAL1=150.0, CRPIX1=1.0, CRVAL2=2.0, CRPIX2=1.0)


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
    beam = ConfigProblem(
        'error', ['beam'], 'BMAJ, BMIN missing (BUNIT Jy/beam)')
    assert third.problems == (ConfigProblem(
        'error', ['spectral', 'convention'], 'VELREF missing'), beam)
    # (its convention given, its velocities are read)
    third = describe_fits(str(tmp_path / 'f.fits'), velocity='radio')[2]
    assert third.coords.convention == 'radio'
    assert third.problems == (beam,)


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
    header = fits.Header(REFERENCES | dict(
        CTYPE1='RA---TAN', CDELT1=-1 / 3600, CTYPE2='DEC--TAN',
        CDELT2=1 / 3600, CTYPE3='WAVELENGTH', CRVAL3=6500, CDELT3=2,
        CRPIX3=1, CUNIT3='Angstrom'))
    fits.writeto(tmp_path / 'califa.fits', np.zeros((4, 3, 2)), header)
    filename = str(tmp_path / 'califa.fits')
    (hdu,) = describe_fits(filename, rest='6562.8 Angstrom')
    assert hdu.coords is None
    assert hdu.problems == (ConfigProblem(
        'error', ['spectral', 'type'], "CTYPE3 'WAVELENGTH' not FITS"),)
    (hdu,) = describe_fits(
        filename, rest='6562.8 Angstrom', spectral_type='AWAV')
    assert hdu.problems == () and hdu.coords.medium == 'air'
    np.testing.assert_allclose(
        hdu.coords.step[2], 299792.458 * 2 / 6562.8, rtol=1e-6)
    # (read as data, as describe reads them; a type not of FITS refused)
    with pytest.raises(ValueError, match="'WAVELENGTH' not FITS"):
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
    assert hdu.problems == () and hdu.coords.medium == 'air'


def test_a_spectral_axis_of_no_type_is_read_as_given(tmp_path):
    # The third axis of a cube of RA and Dec, without CTYPE3 (nor CUNIT3):
    # refused, as nothing says what its values are (not taken as
    # velocities in km/s); of the type given, in m/s (SI, as FITS reads
    # them without a unit), or in the unit given
    header = fits.Header(REFERENCES | dict(
        CTYPE1='RA---TAN', CDELT1=-1 / 3600, CUNIT1='deg',
        CTYPE2='DEC--TAN', CDELT2=1 / 3600, CUNIT2='deg',
        CRVAL3=200.0, CDELT3=4.0, CRPIX3=1))
    fits.writeto(tmp_path / 'untyped.fits', np.zeros((5, 3, 2)), header)
    filename = str(tmp_path / 'untyped.fits')
    (hdu,) = describe_fits(filename)
    assert hdu.problems == (
        ConfigProblem('error', ['spectral', 'type'], 'CTYPE3 missing'),)
    assert hdu.coords is None
    with pytest.raises(ValueError, match='CTYPE3 missing'):
        read_data(filename)
    (hdu,) = describe_fits(filename, spectral_type='VRAD')
    assert hdu.problems == (
        ConfigProblem('warning', ['spectral', 'unit'], 'CUNIT3 missing'),)
    assert hdu.coords.step[2] == pytest.approx(0.004)
    (hdu,) = describe_fits(
        filename, spectral_type='VRAD', spectral_unit='km/s')
    assert hdu.problems == ()
    assert (hdu.coords.step[2], hdu.coords.rval[2]) == (4, 200)
    _, coords = read_data(
        filename, spectral_type='VRAD', spectral_unit='km/s')
    assert (coords.step[2], coords.rval[2]) == (4, 200)
    # (of a unit in the header: read in it once a type is; of a type of
    # another kind, refused)
    header['CUNIT3'] = 'km/s'
    fits.writeto(tmp_path / 'kms.fits', np.zeros((5, 3, 2)), header)
    (hdu,) = describe_fits(str(tmp_path / 'kms.fits'), spectral_type='VRAD')
    assert hdu.problems == () and hdu.coords.step[2] == 4
    (hdu,) = describe_fits(str(tmp_path / 'kms.fits'), spectral_type='AWAV')
    assert hdu.problems == (ConfigProblem(
        'error', ['spectral', 'unit'], "CUNIT3 'km/s' not of AWAV"),)


@pytest.mark.parametrize('cards, unstated', [
    # (no axis of a type: of pixels alone, read as it is)
    (dict(CDELT1=1.0, CDELT2=1.0, CDELT3=4.0), None),
    # (an axis of no type beside a spectral one: a position, as read)
    (dict(CDELT1=1.0, CTYPE2='OFFSET', CDELT2=1.0, CTYPE3='VRAD',
          CUNIT3='km/s', CDELT3=4.0), None),
    # (a type FITS does not define, of no unit and no spectral axis)
    (dict(CTYPE1='RA---TAN', CDELT1=-1 / 3600, CTYPE2='DEC--TAN',
          CDELT2=1 / 3600, CTYPE3='VELOCITY', CDELT3=4.0), (2, 'VELOCITY')),
    # (one of a unit of angles: a position, and the axis of no type the
    # spectral one)
    (dict(CTYPE1='ANGLE', CUNIT1='deg', CDELT1=1 / 3600, CTYPE2='LINEAR',
          CDELT2=1.0, CDELT3=4.0), (2, None))])
def test_which_axis_does_not_say_what_it_is(cards, unstated):
    header = fits.Header(dict(NAXIS=3, NAXIS1=2, NAXIS2=3, NAXIS3=5, **cards))
    assert unstated_spectral_axis(header, (5, 3, 2)) == unstated


def test_a_cube_of_wavelengths_is_described_in_velocities_of_a_rest(
        tmp_path):
    header = fits.Header(REFERENCES | dict(
        CTYPE1='RA---TAN', CDELT1=-0.2 / 3600, CTYPE2='DEC--TAN',
        CDELT2=0.2 / 3600, CTYPE3='AWAV', CRVAL3=6600, CDELT3=1.25,
        CRPIX3=1, CUNIT3='Angstrom'))
    fits.writeto(tmp_path / 'ifu.fits', np.zeros((4, 3, 2)), header)
    (hdu,) = describe_fits(str(tmp_path / 'ifu.fits'))
    assert hdu.coords is None and hdu.problems == (ConfigProblem(
        'error', ['spectral', 'rest'], 'AWAV axis: rest missing'),)
    (hdu,) = describe_fits(str(tmp_path / 'ifu.fits'), rest='6562.8 Angstrom')
    assert hdu.problems == ()
    np.testing.assert_allclose(
        hdu.coords.step, (0.2, 0.2, 299792.458 * 1.25 / 6562.8), rtol=1e-3)


def test_a_spectral_axis_of_no_unit_or_a_bad_one_takes_one_given(
        tmp_path, caplog):
    # Velocities in km/s without a unit, which FITS reads in m/s (with a
    # warning, where data are read); the same with a unit that is not one,
    # refused; wavelengths in Angstrom without a unit; and velocities in
    # m/s without a unit, read rightly: none guessed
    sky = dict(
        CTYPE1='RA---SIN', CRVAL1=150.0, CDELT1=-1 / 3600, CRPIX1=4,
        CTYPE2='DEC--SIN', CRVAL2=2.0, CDELT2=1 / 3600, CRPIX2=3)
    cubes = dict(
        kms=dict(CTYPE3='VRAD', CRVAL3=200.0, CDELT3=4.0),
        bad=dict(CTYPE3='VRAD', CRVAL3=200.0, CDELT3=4.0, CUNIT3='KMS'),
        angstrom=dict(CTYPE3='AWAV', CRVAL3=6650.0, CDELT3=1.0),
        ms=dict(CTYPE3='VRAD', CRVAL3=2e5, CDELT3=4e3))
    for name, spectral in cubes.items():
        fits.PrimaryHDU(np.zeros((5, 6, 7), np.float32), fits.Header(
            sky | spectral | dict(CRPIX3=1))).writeto(tmp_path / f'{name}.f')

    def described(name, **options):
        return describe_fits(str(tmp_path / f'{name}.f'), **options)[0]

    kms = described('kms')
    missing = ConfigProblem('warning', ['spectral', 'unit'], 'CUNIT3 missing')
    assert kms.problems == (missing,)
    assert kms.coords.step[2] == pytest.approx(0.004)
    with caplog.at_level('WARNING'):
        read_data(str(tmp_path / 'kms.f'))
    assert 'CUNIT3 missing: read in m/s (FITS default)' in caplog.text
    caplog.clear()
    kms = described('kms', spectral_unit='km/s')
    assert kms.problems == ()
    assert kms.coords.step[2] == pytest.approx(4)
    assert kms.axis_units[2] == ''
    with caplog.at_level('WARNING'):
        read_data(str(tmp_path / 'kms.f'), spectral_unit='km/s')
    assert 'CUNIT3' not in caplog.text
    bad = described('bad')
    assert bad.coords is None and bad.problems == (ConfigProblem(
        'error', ['spectral', 'unit'], "CUNIT3 'KMS' invalid"),)
    assert bad.axis_units[2] == 'KMS'
    assert described('bad', spectral_unit='km/s').coords.step[2] == \
        pytest.approx(4)
    angstrom = described('angstrom', rest='6562.8 Angstrom')
    assert missing in angstrom.problems
    angstrom = described(
        'angstrom', rest='6562.8 Angstrom', spectral_unit='Angstrom')
    assert angstrom.coords.step[2] == pytest.approx(
        299792.458 / 6562.8, rel=1e-3)
    ms = described('ms')
    assert ms.problems == (missing,)
    assert ms.coords.step[2] == pytest.approx(4)
    # (a unit the header gives, as written)
    fits.PrimaryHDU(np.zeros((5, 6, 7), np.float32), fits.Header(
        sky | cubes['kms'] | dict(CRPIX3=1, CUNIT3='km/s'))).writeto(
            tmp_path / 'given.f')
    assert described('given').problems == ()
    # (a unit that is not of a spectral axis is a problem of the reading)
    jansky = described('kms', spectral_unit='Jy')
    assert jansky.coords is None
    (problem,) = jansky.problems
    assert problem.path == ['spectral', 'unit']
    assert 'velocities, wavelengths or frequencies' in problem.message


def test_independent_problems_of_a_spectral_axis_are_found_together(
        tmp_path):
    # A unit that is not one, and velocities of no convention (as MIRIAD
    # writes them): both at once, each at its option
    fits.writeto(tmp_path / 'cube.fits', np.zeros((4, 3, 2)), fits.Header(
        REFERENCES | dict(
            CTYPE1='RA---TAN', CDELT1=-1 / 3600, CTYPE2='DEC--TAN',
            CDELT2=1 / 3600, CTYPE3='VELO-HEL', CDELT3=4.0, CRVAL3=0.0,
            CRPIX3=1, CUNIT3='KMS')))
    (hdu,) = describe_fits(str(tmp_path / 'cube.fits'))
    assert [(p.path, p.message) for p in hdu.problems] == [
        (['spectral', 'unit'], "CUNIT3 'KMS' invalid"),
        (['spectral', 'convention'], 'VELREF missing')]
    (hdu,) = describe_fits(
        str(tmp_path / 'cube.fits'), velocity='radio', spectral_unit='km/s')
    assert hdu.problems == () and hdu.coords.step[2] == 4
