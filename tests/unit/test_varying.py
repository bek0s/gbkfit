"""
Tests for the PSFs and LSFs that vary along the spectral axis.
"""

import astropy.table
import astropy.units as u
import numpy as np
import pytest

from gbkfit.config import ConfigError, paths_relative_to
from gbkfit.grid import VelocityConvention
from gbkfit.instrument import lsf_parser, psf_parser


C = 299792.458
KMS = u.km / u.s
WAVELENGTHS = [4800.0, 7000.0, 9300.0]
ALPHAS = [0.75, 0.65, 0.58]


def alpha_table(tmp_path, source):
    """The table of alpha against wavelength, from the given source."""
    if source == 'inline':
        return dict(
            points=dict(value=WAVELENGTHS, unit='Angstrom'), values=ALPHAS)
    if source == 'csv':
        # (a plain CSV file has no units)
        path = tmp_path / 'psf.csv'
        path.write_text('wavelength,alpha\n' + ''.join(
            f'{w},{a}\n' for w, a in zip(WAVELENGTHS, ALPHAS)))
        return dict(
            file=str(path), points='wavelength', column='alpha',
            units=dict(wavelength='Angstrom'))
    path = tmp_path / f'psf.{source}'
    astropy.table.QTable(dict(
        wavelength=WAVELENGTHS * u.AA, alpha=ALPHAS * u.arcsec)).write(path)
    return dict(file=str(path), points='wavelength', column='alpha')


@pytest.mark.parametrize('source', ['inline', 'csv', 'ecsv', 'fits'])
def test_psf_options_vary_with_wavelength(tmp_path, source):
    # The velocities of an axis of a rest wavelength are optical
    psf = psf_parser.load(dict(
        type='moffat', alpha=alpha_table(tmp_path, source), beta=2.5))
    assert psf.varies()
    rest = 6562.8 * u.AA
    velocities = np.array([-100000.0, 0.0, 20000.0, 200000.0])
    psfs = psf.at(velocities * KMS, VelocityConvention(rest))
    wavelengths = 6562.8 * (1 + velocities / C)
    np.testing.assert_allclose(
        [p.dump()['alpha'] for p in psfs],
        np.interp(wavelengths, WAVELENGTHS, ALPHAS), rtol=1e-9)
    assert {p.dump()['beta'] for p in psfs} == {2.5}
    # (the first and the last are beyond the table: its nearest values)
    assert [psfs[0].dump()['alpha'], psfs[-1].dump()['alpha']] == [
        ALPHAS[0], ALPHAS[-1]]


def test_beams_vary_with_frequency():
    # The velocities of an axis of a rest frequency are radio
    beam = psf_parser.load(dict(
        type='gauss', bmin=1.0,
        bmaj=dict(points=dict(value=[1419.0, 1421.0], unit='MHz'),
                  values=[3.0, 2.0])))
    rest = 1420.405752 * u.MHz
    velocities = np.array([0.0, 100.0])
    frequencies = 1420.405752 * (1 - velocities / C)
    np.testing.assert_allclose(
        [b.dump()['bmaj'] for b in beam.at(
            velocities * KMS, VelocityConvention(rest))],
        np.interp(frequencies, [1419.0, 1421.0], [3.0, 2.0]), rtol=1e-9)


def test_lsf_widths_keep_the_unit_of_their_table():
    # Interpolated in Angstrom, a width that spans c dw / rest km/s of an
    # axis of optical velocities at each channel (see Channel.span)
    from gbkfit.instrument import Channel
    lsf = lsf_parser.load(dict(type='gauss', sigma=dict(
        points=dict(value=[-100, 100], unit='km/s'), values=[1.0, 2.0],
        unit='Angstrom')))
    convention = VelocityConvention(6562.8 * u.AA)
    velocities = np.array([0.0, 100.0])
    lsfs = lsf.at(velocities * KMS, convention)
    assert [x.dump()['sigma'] for x in lsfs] == [
        '1.5 Angstrom', '2.0 Angstrom']
    np.testing.assert_allclose(
        [Channel(v, None, convention).span(x._scale)
         for x, v in zip(lsfs, velocities)],
        C * np.array([1.5, 2.0]) / 6562.8, rtol=1e-12)


def test_sums_of_varying_psfs_vary():
    table = dict(
        points=dict(value=[0, 100, 200], unit='km/s'), values=[1.0, 2.0, 3.0])
    psf = psf_parser.load(dict(type='sum', weights=[1, 1], psfs=[
        dict(type='gauss', sigma=table),
        dict(type='gauss', sigma=dict(
            points=dict(value=[50, 300], unit='km/s'), values=[1, 1]))]))
    assert psf.varies()
    assert psf.velocity_range() == (50, 200)
    sums = psf.at([150] * KMS)
    assert sums[0].dump()['psfs'][0]['sigma'] == 2.5
    # A sum of constant PSFs does not vary
    constant = psf_parser.load(dict(type='sum', weights=[1], psfs=[
        dict(type='gauss', sigma=1)]))
    assert not constant.varies()
    assert constant.at([0, 1] * KMS) == [constant, constant]


def test_varying_psfs_round_trip():
    info = dict(
        type='gauss', ratio=0.5,
        sigma=dict(points=dict(value=[5000.0, 6000.0], unit='Angstrom'),
                   values=[1.0, 2.0]),
        posa=dict(points=dict(value=[0.0, 10.0], unit='km / s'),
                  values=[0.1, 0.2], unit='rad'))
    psf = psf_parser.load(dict(info))
    dumped = psf_parser.dump(psf)
    assert psf_parser.dump(psf_parser.load(dict(dumped))) == dumped
    # (an angle in radians is converted to the degrees of posa)
    np.testing.assert_allclose(
        psf.at([10] * KMS, VelocityConvention(6000 * u.AA))[0].dump()[
            'posa'],
        np.degrees(0.2))


def kms(values):
    """The configuration of the points of a table at the given velocities."""
    return dict(value=values, unit='km/s')


@pytest.mark.parametrize('info, error, message', [
    (dict(type='hanning', channels=dict(points=kms([0]), values=[1])),
     ConfigError, "option 'channels' must be of type float"),
    # Of the values and the points of tables, which the constructors check
    (dict(type='gauss', sigma=dict(
        points=kms([0, 1]), values=[1, -1], unit='km/s')),
     ValueError, "at 1.0 km s-1: option 'sigma' must be greater than 0"),
    (dict(type='gauss', sigma=dict(points=kms([0, 1]), values=[1, 2])),
     ValueError, "option 'sigma' of LSFGauss has a unit"),
    (dict(type='gauss', sigma=dict(points=[1, 2], values=[1, 1])),
     ConfigError, "sigma: option 'points' must be of type Quantity"),
    (dict(type='gauss', sigma=dict(points=kms([0, 0]), values=[1, 1])),
     ValueError, "must be distinct"),
    # (a unit that the option cannot be in: refused as the LSF is made,
    # though its points are velocities and wavelengths, which only the
    # data put together)
    (dict(type='moffat',
          alpha=dict(points=kms([0, 100]), values=[1, 2], unit='km/s'),
          beta=dict(points=dict(value=[6000, 7000], unit='Angstrom'),
                    values=[2, 3], unit='arcsec')),
     ValueError, "'arcsec', which cannot be converted to ''")])
def test_varying_options_are_checked(info, error, message):
    with pytest.raises(error, match=message):
        lsf_parser.load(info)


def test_an_lsf_that_varies_is_drawn_as_it_is_at_each_channel():
    from gbkfit.instrument import Channel
    lsf = lsf_parser.load(dict(type='gauss', sigma=dict(
        points=kms([0, 100]), values=[10, 20], unit='km/s')))
    with pytest.raises(RuntimeError, match="as it is at each point"):
        lsf.on(Channel())
    lsf.at([50] * KMS)[0].on(Channel()).asarray(1.0)


def test_tables_in_wavelength_need_the_rest():
    psf = psf_parser.load(
        dict(type='gauss', sigma=alpha_table(None, 'inline')))
    with pytest.raises(ValueError, match="convention of the velocities"):
        psf.at([0] * KMS)


def gauss_images(sigmas, size=21):
    """Images of Gaussian PSFs of the given sigmas (pixels)."""
    from gbkfit.instrument import PSFGauss
    return np.stack([
        PSFGauss(sigma).asarray((1, 1), (size, size)) for sigma in sigmas])


def test_psf_images_are_interpolated():
    from gbkfit.instrument import PSFImages, PSFImage
    images = gauss_images([1, 2])
    cube = PSFImages(images, [100, 300] * u.km / u.s, (1, 1))
    assert cube.varies() and cube.velocity_range() == (100, 300)
    expected = [images[0], 0.75 * images[0] + 0.25 * images[1], images[1]]
    for psf, image in zip(cube.at([0, 150, 300] * KMS), expected):
        np.testing.assert_allclose(
            psf.asarray((1, 1), (21, 21)),
            PSFImage(image, (1, 1)).asarray((1, 1), (21, 21)), atol=1e-12)


def test_psf_images_from_files(tmp_path):
    # The points come from the spectral axis of the file, and the size of
    # its pixels from its celestial axes
    import astropy.io.fits
    import astropy.wcs
    wcs = astropy.wcs.WCS(naxis=3)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'WAVE']
    wcs.wcs.cunit = ['deg', 'deg', 'm']
    wcs.wcs.cdelt = [-0.05 / 3600, 0.05 / 3600, 1e-8]
    wcs.wcs.crval = [150, 2, 6.5e-7]
    wcs.wcs.crpix = [11, 11, 1]
    path = tmp_path / 'psf_images.fits'
    astropy.io.fits.writeto(path, gauss_images([1, 2]), wcs.to_header())
    psf = psf_parser.load(dict(type='images', file=str(path)))
    dumped = psf_parser.dump(psf, prefix=str(tmp_path / 'd_'))
    np.testing.assert_allclose(dumped['step'], [0.05, 0.05])
    assert dumped['points']['unit'] == 'm'
    np.testing.assert_allclose(dumped['points']['value'], [6.5e-7, 6.6e-7])
    again = psf_parser.load(dict(dumped))
    rest = 6.5e-7 * u.m
    convention = VelocityConvention(rest)
    for a, b in zip(psf.at([0, 1e3, 1e4] * KMS, convention),
                    again.at([0, 1e3, 1e4] * KMS, convention)):
        np.testing.assert_allclose(
            a.asarray((0.05, 0.05)), b.asarray((0.05, 0.05)))


def test_lsf_profiles_from_files(tmp_path):
    # One profile per row: the points come from the spectral axis of the
    # rows, and the width of the channels from the columns (km/s)
    import astropy.io.fits
    import astropy.wcs
    from gbkfit.instrument import Channel, LSFGauss, LSFProfile
    profiles = np.stack([
        LSFGauss(sigma * KMS).on(Channel()).asarray(5, 41)
        for sigma in (20, 40)])
    wcs = astropy.wcs.WCS(naxis=2)
    wcs.wcs.ctype = ['', 'WAVE']
    wcs.wcs.cunit = ['km/s', 'Angstrom']
    wcs.wcs.cdelt = [5, 2000]
    wcs.wcs.crval = [0, 5000]
    wcs.wcs.crpix = [21, 1]
    path = tmp_path / 'lsf_profiles.fits'
    astropy.io.fits.writeto(path, profiles, wcs.to_header())
    lsf = lsf_parser.load(dict(type='profiles', file=str(path)))
    assert lsf.varies()
    dumped = lsf_parser.dump(lsf, prefix=str(tmp_path / 'd_'))
    assert dumped['step'] == '5.0 km s-1'
    points = dumped['points']
    np.testing.assert_allclose(
        u.Quantity(points['value'], points['unit']).to_value(u.AA),
        [5000, 7000])
    # Halfway between the points, half of each profile
    middle, = lsf_parser.load(dict(dumped)).at([6000] * u.AA)
    np.testing.assert_allclose(
        middle.on(Channel()).asarray(5, 41),
        LSFProfile((profiles[0] + profiles[1]) / 2, 5 * KMS).on(
            Channel()).asarray(5, 41),
        atol=1e-12)


def test_varying_psfs_in_python(tmp_path):
    # The same PSF as from its configuration, queried at wavelengths
    # (with no rest, as the table is in wavelength) or velocities
    from gbkfit.instrument import PSFImage, PSFMoffat, SpectralTable
    alpha = SpectralTable(WAVELENGTHS * u.AA, ALPHAS)
    psf = PSFMoffat.varying(alpha=alpha, beta=2.5)
    loaded = psf_parser.load(dict(
        type='moffat', alpha=alpha_table(tmp_path, 'inline'), beta=2.5))
    assert psf_parser.dump(psf) == psf_parser.dump(loaded)
    assert psf.at([5900] * u.AA)[0].dump()['alpha'] == pytest.approx(0.7)
    velocity = (5900 * u.AA).to(KMS, u.doppler_optical(6000 * u.AA))
    convention = VelocityConvention(6000 * u.AA)
    assert psf.at([velocity], convention)[0].dump()['alpha'] == (
        pytest.approx(0.7))
    # (only the analytic ones vary by tables of their options)
    assert not hasattr(PSFImage, 'varying')
    with pytest.raises(ValueError, match="none of the options of PSFMoffat"):
        PSFMoffat.varying(alpha=0.7, beta=2.5)


def test_spectral_tables_from_files(tmp_path):
    from gbkfit.instrument import SpectralTable
    table = SpectralTable.from_file(
        alpha_table(tmp_path, 'ecsv')['file'], 'wavelength', 'alpha')
    np.testing.assert_allclose(table.points().to_value(u.AA), WAVELENGTHS)
    np.testing.assert_allclose(table.values(), ALPHAS)
    assert table.unit() == u.arcsec


def test_a_masked_value_of_a_table_file_is_refused(tmp_path):
    # (it was read as 0)
    from gbkfit.instrument import SpectralTable
    table = astropy.table.Table(dict(wavelength=WAVELENGTHS * u.AA))
    table['alpha'] = astropy.table.MaskedColumn(
        ALPHAS, mask=[i == 1 for i in range(len(ALPHAS))])
    table.write(str(tmp_path / 'psf.ecsv'))
    with pytest.raises(ValueError, match="must be finite"):
        SpectralTable.from_file(
            str(tmp_path / 'psf.ecsv'), 'wavelength', 'alpha')


def test_the_columns_of_a_table_file_are_named_by_the_configuration(
        tmp_path):
    # (any name, e.g. of FITS in capitals; the unit says what the points
    # are, and they need one)
    from gbkfit.instrument import SpectralTable
    path = str(tmp_path / 'psf.fits')
    astropy.table.QTable(dict(
        WAVE=WAVELENGTHS * u.AA, ALPHA=ALPHAS * u.arcsec)).write(path)
    table = SpectralTable.from_file(path, 'WAVE', 'ALPHA')
    np.testing.assert_allclose(table.points().to_value(u.AA), WAVELENGTHS)
    with pytest.raises(ValueError, match="no column 'wavelength'"):
        SpectralTable.from_file(path, 'wavelength', 'ALPHA')
    astropy.table.Table(dict(WAVE=WAVELENGTHS, ALPHA=ALPHAS)).write(
        path, overwrite=True)
    with pytest.raises(ValueError, match="'WAVE', has no unit"):
        SpectralTable.from_file(path, 'WAVE', 'ALPHA')


def test_table_files_are_relative_to_the_configuration(tmp_path):
    # A relative table file is in the folder of the configuration (see
    # paths_relative_to), not in the current one
    folder = tmp_path / 'config'
    folder.mkdir()
    info = alpha_table(folder, 'ecsv') | dict(file='psf.ecsv')
    with paths_relative_to(str(folder)):
        psf = psf_parser.load(dict(type='moffat', alpha=info, beta=2.5))
    assert psf.at([5900] * u.AA)[0].dump()['alpha'] == pytest.approx(0.7)


def test_tables_are_checked_together_at_their_points():
    # The beams of a cube of CASA, nearly round and drifting across its
    # band, are valid at each channel, though the bmin of one channel is
    # above the bmaj of another
    ghz = dict(value=[1.0, 1.1, 1.2], unit='GHz')
    psf = psf_parser.load(dict(
        type='gauss', bmaj=dict(points=ghz, values=[10.0, 9.9995, 9.999]),
        bmin=dict(points=ghz, values=[9.9994, 9.999, 9.998])))
    assert psf.at([1.2] * u.GHz)[0].dump()['bmin'] == 9.998
    # Values are compared in the units of the options
    psf_parser.load(dict(
        type='gauss', bmaj=dict(points=ghz, values=[10.0, 10.0, 10.0]),
        bmin=dict(points=ghz, values=[9000, 9500, 9900], unit='mas')))
    # Values invalid together are refused, at the point where they are
    with pytest.raises(
            ValueError, match="at 1.1 GHz: bmin must be at most bmaj"):
        psf_parser.load(dict(
            type='gauss',
            bmaj=dict(points=ghz, values=[10.0, 10.0, 10.0]),
            bmin=dict(points=dict(value=[1.0, 1.15, 1.2], unit='GHz'),
                      values=[9.0, 11.0, 9.0])))


def test_lsfs_vary_by_their_fwhm():
    # A table of the FWHM, as specifications give it (e.g. that of MUSE),
    # and of the shape: the profile of each at each channel
    from gbkfit.instrument import LSFMoffat
    lsf = lsf_parser.load(dict(
        type='moffat',
        fwhm=dict(points=dict(value=[5000, 9000], unit='Angstrom'),
                  values=[2.8, 2.4], unit='Angstrom'),
        beta=dict(points=dict(value=[5000, 9000], unit='Angstrom'),
                  values=[2.0, 3.0])))
    at, = lsf.at([7000] * u.AA)
    expected = LSFMoffat.from_fwhm(2.6 * u.AA, 2.5)
    assert at.dump()['beta'] == 2.5
    from gbkfit.units import parse_quantity
    assert u.isclose(parse_quantity(at.dump()['fwhm']), 2.6 * u.AA)
    assert u.isclose(at._scale, expected._scale)
    assert lsf_parser.dump(lsf)['fwhm']['unit'] == 'Angstrom'
