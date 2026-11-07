"""
Tests for the reading of data with their errors and mask.
"""

import astropy.units
import astropy.wcs
import numpy as np
import pytest
from astropy.io import fits

from gbkfit.config import ConfigError
from gbkfit.dataset import describe_fits
from gbkfit.io import coords_from_header, read_array, read_arrays, read_data


# A cube of 5 channels of radio velocities that decrease along the file,
# from 1000 to 960 km/s
CUBE = {
    'CTYPE1': 'RA---TAN', 'CRVAL1': 150.0, 'CRPIX1': 3.0,
    'CDELT1': -1 / 3600, 'CUNIT1': 'deg',
    'CTYPE2': 'DEC--TAN', 'CRVAL2': 2.0, 'CRPIX2': 2.0,
    'CDELT2': 1 / 3600, 'CUNIT2': 'deg',
    'CTYPE3': 'VRAD', 'CRVAL3': 1000.0, 'CRPIX3': 1.0,
    'CDELT3': -10.0, 'CUNIT3': 'km/s', 'BUNIT': 'Jy/beam'}
SHAPE = (5, 3, 4)


def write(filename, data, header=None):
    fits.writeto(
        filename, np.asarray(data, np.float32), fits.Header(header or {}))


@pytest.mark.parametrize('kind, values, unit', [
    ('sigma', 2.0, None),
    ('variance', 4.0, None),
    ('inverse_variance', 0.25, None),
    ('sigma', 2000.0, 'mJy/beam'),
    ('variance', 4e6, 'mJy2 / beam2')])
def test_errors_of_each_kind_become_standard_deviations(
        tmp_path, kind, values, unit):
    # In the unit of the data, whatever theirs
    write(tmp_path / 'data.fits', np.ones(SHAPE), CUBE)
    write(tmp_path / 'error.fits', np.full(SHAPE, values),
          {} if unit is None else dict(BUNIT=unit))
    arrays = read_arrays(
        str(tmp_path / 'data.fits'), str(tmp_path / 'error.fits'),
        error_kind=kind)
    np.testing.assert_allclose(arrays.error, 2.0, rtol=1e-6)


def test_errors_in_the_unit_of_another_kind_are_an_error(tmp_path):
    write(tmp_path / 'data.fits', np.ones(SHAPE), CUBE)
    write(tmp_path / 'error.fits', np.full(SHAPE, 4.0),
          dict(BUNIT='Jy2 / beam2'))
    with pytest.raises(ValueError, match="error_kind: variance"):
        read_arrays(
            str(tmp_path / 'data.fits'), str(tmp_path / 'error.fits'))


def test_inverse_variances_of_zero_are_values_without_data(tmp_path):
    write(tmp_path / 'data.fits', np.ones(SHAPE), CUBE)
    inverse = np.full(SHAPE, 4.0)
    inverse[0, 0, 0] = 0
    write(tmp_path / 'error.fits', inverse)
    arrays = read_arrays(
        str(tmp_path / 'data.fits'), str(tmp_path / 'error.fits'),
        error_kind='inverse_variance')
    assert arrays.mask.sum() == arrays.mask.size - 1
    assert np.isnan(arrays.data[-1, 0, 0])


def test_errors_follow_their_own_world_coordinates(tmp_path):
    # The data decrease in velocity along their file, the errors (each
    # the velocity of its channel) increase along theirs: both read in
    # increasing velocity, each channel keeps its error; without world
    # coordinates, the errors follow the axes of the file of the data
    velocities = 1000 - 10 * np.arange(SHAPE[0])
    write(tmp_path / 'data.fits',
          np.broadcast_to(velocities[:, None, None], SHAPE), CUBE)
    write(tmp_path / 'error.fits',
          np.broadcast_to(velocities[::-1, None, None], SHAPE),
          CUBE | dict(CRVAL3=960.0, CDELT3=10.0))
    write(tmp_path / 'plain.fits',
          np.broadcast_to(velocities[:, None, None], SHAPE))
    for error in ('error.fits', 'plain.fits'):
        arrays = read_arrays(
            str(tmp_path / 'data.fits'), str(tmp_path / error))
        np.testing.assert_array_equal(arrays.data[:, 0, 0], velocities[::-1])
        np.testing.assert_array_equal(arrays.error, arrays.data)


def test_a_footprint_follows_the_data(tmp_path):
    # Without world coordinates, in the axes of the file of the data,
    # whose velocities decrease along it: read in increasing velocity,
    # as the data (here observed in the first channel of the file only)
    write(tmp_path / 'data.fits', np.ones(SHAPE), CUBE)
    footprint = np.zeros(SHAPE)
    footprint[0] = 1
    write(tmp_path / 'footprint.fits', footprint)
    arrays = read_arrays(
        str(tmp_path / 'data.fits'),
        footprint=str(tmp_path / 'footprint.fits'))
    np.testing.assert_array_equal(arrays.footprint, footprint[::-1])


def test_errors_on_other_pixels_are_an_error(tmp_path):
    write(tmp_path / 'data.fits', np.ones(SHAPE), CUBE)
    write(tmp_path / 'error.fits', np.ones(SHAPE), CUBE | dict(CRVAL3=990.0))
    with pytest.raises(ValueError, match="pixels"):
        read_arrays(
            str(tmp_path / 'data.fits'), str(tmp_path / 'error.fits'))


def test_the_beams_of_the_channels_follow_them(tmp_path):
    # A cube of CASA with a beam for each channel (its table BEAMS, and no
    # BMAJ), whose velocities decrease along the file: read with its
    # channels reversed, its beams are in the order of its channels, as a
    # dataset, and for the fluxes of its pixels (of 1 arcsec^2, over the
    # area of the beam of their channel); it is described
    from gbkfit.dataset import Data
    from gbkfit.prep import FitsArrays, pixel_fluxes
    bmaj = np.array([10.0, 11, 12, 13, 14])
    beams = fits.BinTableHDU.from_columns([
        fits.Column(name='BMAJ', format='E', unit='arcsec', array=bmaj),
        fits.Column(name='BMIN', format='E', unit='arcsec', array=[5] * 5),
        fits.Column(name='BPA', format='E', unit='deg', array=[0] * 5),
        fits.Column(name='CHAN', format='J', array=range(5)),
        fits.Column(name='POL', format='J', array=[0] * 5)], name='BEAMS')
    path = str(tmp_path / 'cube.fits')
    fits.HDUList([
        fits.PrimaryHDU(
            (np.arange(5.0)[:, None, None] * np.ones(SHAPE)).astype(
                np.float32), fits.Header(CUBE | dict(CASAMBM=True))),
        beams]).writeto(path)
    arrays = read_arrays(path)
    np.testing.assert_array_equal(arrays.data[:, 0, 0], [4, 3, 2, 1, 0])
    np.testing.assert_allclose(arrays.beam['bmaj'], bmaj[::-1])
    np.testing.assert_allclose(
        arrays.beam['spectral'].to_value('km/s'), [960, 970, 980, 990, 1000])
    data, _ = Data.from_files(path)
    np.testing.assert_allclose(data.beam().bmaj, bmaj[::-1])
    cube = FitsArrays.read(path).crop(
        (slice(1, 4), slice(None), slice(None)))
    np.testing.assert_allclose(cube.beam['bmaj'], [13, 12, 11])
    factors, _ = pixel_fluxes(cube.header, cube.beam, path)
    np.testing.assert_allclose(
        factors, 1 / (np.pi / (4 * np.log(2)) * np.array([13, 12, 11]) * 5),
        rtol=1e-6)
    assert describe_fits(path)[0].beam is not None


def test_the_beams_follow_the_channels_past_an_axis_of_stokes(tmp_path):
    # The axes of a cube of CASA: RA, Dec, Stokes (of 1) and increasing
    # frequency, whose radio velocities of a rest decrease, so its channels
    # are reversed; the spectral axis is the third of the model, the fourth
    # of the file
    header = fits.Header(dict(
        CTYPE1='RA---SIN', CRVAL1=10.0, CDELT1=-1 / 3600, CRPIX1=3.0,
        CUNIT1='deg', CTYPE2='DEC--SIN', CRVAL2=-30.0, CDELT2=1 / 3600,
        CRPIX2=3.0, CUNIT2='deg', CTYPE3='STOKES', CRVAL3=1.0, CDELT3=1.0,
        CRPIX3=1.0, CTYPE4='FREQ', CRVAL4=1.42e9, CDELT4=1e5, CRPIX4=1.0,
        CUNIT4='Hz', SPECSYS='LSRK'))
    values = np.arange(4.0)[:, None, None, None] * np.ones((4, 1, 5, 6))
    beams = fits.BinTableHDU.from_columns([
        fits.Column(name='BMAJ', format='E', unit='arcsec',
                    array=[1, 2, 3, 4]),
        fits.Column(name='BMIN', format='E', unit='arcsec', array=[1] * 4),
        fits.Column(name='BPA', format='E', unit='deg', array=[0] * 4),
        fits.Column(name='CHAN', format='J', array=range(4)),
        fits.Column(name='POL', format='J', array=[0] * 4)], name='BEAMS')
    path = str(tmp_path / 'cube.fits')
    fits.HDUList([
        fits.PrimaryHDU(values.astype(np.float32), header), beams
    ]).writeto(path)
    arrays = read_arrays(path, rest='1.420405752 GHz')
    np.testing.assert_array_equal(arrays.data[:, 0, 0], [3, 2, 1, 0])
    np.testing.assert_allclose(arrays.beam['bmaj'], [4, 3, 2, 1])


def test_a_bunit_that_is_not_a_unit_is_an_error_unless_a_unit_is_given(
        tmp_path):
    # (the errors are then in the unit given, as without BUNIT)
    for name in ('data', 'error'):
        write(tmp_path / f'{name}.fits', np.ones(SHAPE),
              CUBE | dict(BUNIT='Jy/bm'))
    data, error = str(tmp_path / 'data.fits'), str(tmp_path / 'error.fits')
    with pytest.raises(ValueError, match="BUNIT = 'Jy/bm' is not a unit"):
        read_arrays(data)
    arrays = read_arrays(data, error, unit='Jy/beam')
    assert arrays.unit == astropy.units.Jy / astropy.units.beam
    np.testing.assert_allclose(arrays.error, 1)


def test_world_axes_beyond_the_array_are_flat(tmp_path):
    # A moment map written with the header of its cube (astropy keeps its
    # spectral axis) reads as a map
    filename = str(tmp_path / 'moment.fits')
    write(filename, np.ones(SHAPE[1:]), CUBE | dict(BMAJ=1e-3, BMIN=1e-3))
    data, coords = read_data(filename)
    assert data.shape == SHAPE[1:] and len(coords.step) == 2
    assert read_array(filename)[0].shape == SHAPE[1:]
    description, = describe_fits(filename)
    assert description.problems == ()


# A cube of the sky (0.1 arcsec pixels, rotated by 30 degrees) and of
# wavelengths about Halpha at z = 1.5 (micron)
TWIN_SHAPE = (6, 5, 7)
TWIN_CD = 0.1 / 3600 * np.array([
    [-np.cos(np.radians(30)), np.sin(np.radians(30))],
    [np.sin(np.radians(30)), np.cos(np.radians(30))]])


def twin_files(tmp_path, kind):
    """
    Write a cube in a FITS file and in an ASDF file, whose world
    coordinates are a gwcs: the wavelengths linear (WAVE) or exponential
    (WAVE-LOG) in the pixels; or, of the ASDF file only, quadratic
    ('curved').
    """
    import asdf
    import astropy.units as u
    from astropy.coordinates import ICRS
    from astropy.modeling import models
    from gwcs import WCS, coordinate_frames as cf
    cube = np.random.default_rng(1).random(TWIN_SHAPE)
    header = dict(
        CTYPE1='RA---TAN', CUNIT1='deg', CRVAL1=150.0, CRPIX1=4.0,
        CTYPE2='DEC--TAN', CUNIT2='deg', CRVAL2=2.0, CRPIX2=3.0,
        CD1_1=TWIN_CD[0, 0], CD1_2=TWIN_CD[0, 1],
        CD2_1=TWIN_CD[1, 0], CD2_2=TWIN_CD[1, 1],
        CTYPE3='WAVE-LOG' if kind == 'WAVE-LOG' else 'WAVE', CUNIT3='um',
        CRVAL3=1.641, CRPIX3=3.0, CD3_3=2e-4,
        BUNIT='Jy')
    fits.writeto(tmp_path / 'cube.fits', cube, fits.Header(header))
    sky = ((models.Shift(-3) & models.Shift(-2))
           | models.AffineTransformation2D(matrix=TWIN_CD * 1.0)
           | models.Pix2Sky_TAN()
           | models.RotateNative2Celestial(150, 2, 180))
    spectral = models.Shift(-2) | dict(
        WAVE=models.Scale(2e-4) | models.Shift(1.641),
        curved=models.Polynomial1D(2, c0=1.641, c1=2e-4, c2=2e-5),
        **{'WAVE-LOG': models.Scale(2e-4) | models.Exponential1D(
            amplitude=1.641, tau=1.641)})[kind]
    wcs = WCS([
        (cf.CoordinateFrame(
            naxes=3, axes_type=('SPATIAL', 'SPATIAL', 'SPECTRAL'),
            axes_order=(0, 1, 2), name='detector', unit=(u.pix,) * 3),
         sky & spectral),
        (cf.CompositeFrame([
            cf.CelestialFrame(
                reference_frame=ICRS(), axes_order=(0, 1),
                unit=(u.deg, u.deg)),
            cf.SpectralFrame(
                axes_order=(2,), unit=(u.um,),
                axes_names=('wavelength',))]), None)])
    asdf.AsdfFile(dict(roman=dict(
        data=cube * u.Jy, err=np.full(TWIN_SHAPE, 0.1) * u.Jy,
        meta=dict(wcs=wcs)))).write_to(tmp_path / 'cube.asdf')
    return str(tmp_path / 'cube.fits'), str(tmp_path / 'cube.asdf')


@pytest.mark.parametrize('kind', ['WAVE', 'WAVE-LOG'])
def test_asdf_files_read_as_their_fits_twins(tmp_path, kind):
    # One reader past the file: the same values, unit and world
    # coordinates (here as the velocities of the rest of Halpha)
    fits_file, asdf_file = twin_files(tmp_path, kind)
    rest = '6564.61 Angstrom'
    from_fits = read_arrays(fits_file, rest=rest)
    from_asdf = read_arrays(
        (asdf_file, 'roman/data'), (asdf_file, 'roman/err'), rest=rest)
    np.testing.assert_array_equal(from_asdf.data, from_fits.data)
    np.testing.assert_allclose(from_asdf.error, 0.1)
    assert from_asdf.unit == from_fits.unit
    pixels = np.array(np.meshgrid(
        [0, 6], [0, 4], [0, 5])).reshape(3, -1)
    world = [
        fits_wcs.pixel_to_world_values(*pixels)
        for fits_wcs in (
            astropy.wcs.WCS(from_fits.header),
            astropy.wcs.WCS(from_asdf.header))]
    np.testing.assert_allclose(world[1], world[0], rtol=1e-9)
    coords = [
        coords_from_header('cube', arrays.header)
        for arrays in (from_fits, from_asdf)]
    np.testing.assert_allclose(coords[1].step, coords[0].step, rtol=1e-6)
    assert coords[1].rota == pytest.approx(coords[0].rota)
    assert coords[1].convention == coords[0].convention


def test_asdf_files_off_a_grid_are_refused(tmp_path):
    # Wavelengths of a curved axis: no linear header stands for them
    _, asdf_file = twin_files(tmp_path, 'curved')
    with pytest.raises(ValueError, match="resample"):
        read_arrays((asdf_file, 'roman/data'))


def test_datasets_read_asdf_files(tmp_path):
    from gbkfit.dataset import dataset_parser
    _, asdf_file = twin_files(tmp_path, 'WAVE')
    dataset = dataset_parser.load(dict(
        type='pixel_spectra',
        data=dict(file=asdf_file, path='roman/data'),
        error=dict(file=asdf_file, path='roman/err'),
        spectral=dict(rest='6564.61 Angstrom')))
    assert dataset['spectra'].shape() == TWIN_SHAPE
    with pytest.raises(ConfigError, match="by its path"):
        dataset_parser.load(dict(
            type='pixel_spectra', data=dict(file=asdf_file, hdu=1)))
