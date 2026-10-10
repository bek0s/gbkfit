"""
Tests for the PSFs and LSFs that vary along the spectral axis.
"""

import astropy.table
import astropy.units as u
import numpy as np
import pytest

from gbkfit.instrument import lsf_parser, psf_parser
from gbkfit.utils.parseutils import ConfigError


C = 299792.458
WAVELENGTHS = [4800.0, 7000.0, 9300.0]
ALPHAS = [0.75, 0.65, 0.58]


def alpha_table(tmp_path, source):
    """The table of alpha against wavelength, from the given source."""
    if source == 'inline':
        return dict(
            wavelength=dict(values=WAVELENGTHS, unit='Angstrom'),
            values=ALPHAS)
    if source == 'csv':
        # (a plain CSV file has no units)
        path = tmp_path / 'psf.csv'
        path.write_text('wavelength,alpha\n' + ''.join(
            f'{w},{a}\n' for w, a in zip(WAVELENGTHS, ALPHAS)))
        return dict(
            table=str(path), column='alpha',
            units=dict(wavelength='Angstrom'))
    path = tmp_path / f'psf.{source}'
    astropy.table.QTable(dict(
        wavelength=WAVELENGTHS * u.AA, alpha=ALPHAS * u.arcsec)).write(path)
    return dict(table=str(path), column='alpha')


@pytest.mark.parametrize('source', ['inline', 'csv', 'ecsv', 'fits'])
def test_psf_options_vary_with_wavelength(tmp_path, source):
    # The velocities of an axis of a rest wavelength are optical
    psf = psf_parser.load(dict(
        type='moffat', alpha=alpha_table(tmp_path, source), beta=2.5))
    assert psf.varies()
    rest = 6562.8 * u.AA
    velocities = np.array([-100000.0, 0.0, 20000.0, 200000.0])
    psfs = psf.at_velocities(velocities, rest)
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
        type='gauss_beam', bmin=1.0,
        bmaj=dict(frequency=dict(values=[1419.0, 1421.0], unit='MHz'),
                  values=[3.0, 2.0])))
    rest = 1420.405752 * u.MHz
    velocities = np.array([0.0, 100.0])
    frequencies = 1420.405752 * (1 - velocities / C)
    np.testing.assert_allclose(
        [b.dump()['bmaj'] for b in beam.at_velocities(velocities, rest)],
        np.interp(frequencies, [1419.0, 1421.0], [3.0, 2.0]), rtol=1e-9)


def test_lsf_widths_in_wavelength_are_converted_to_velocities():
    # With the optical convention, a width of dw is c dw / rest
    lsf = lsf_parser.load(dict(type='gauss', sigma=dict(
        velocity=[-100, 100],
        values=dict(values=[1.0, 2.0], unit='Angstrom'))))
    rest = 6562.8 * u.AA
    sigmas = [x.dump()['sigma'] for x in lsf.at_velocities([0, 100], rest)]
    np.testing.assert_allclose(sigmas, C * np.array([1.5, 2.0]) / 6562.8)


def test_sums_of_varying_psfs_vary():
    table = dict(velocity=[0, 100, 200], values=[1.0, 2.0, 3.0])
    psf = psf_parser.load(dict(type='sum', weights=[1, 1], psfs=[
        dict(type='gauss', sigma=table),
        dict(type='gauss', sigma=dict(velocity=[50, 300], values=[1, 1]))]))
    assert psf.varies()
    assert psf.velocity_range() == (50, 200)
    sums = psf.at_velocities([150])
    assert sums[0].dump()['psfs'][0]['sigma'] == 2.5
    # A sum of constant PSFs does not vary
    constant = psf_parser.load(dict(type='sum', weights=[1], psfs=[
        dict(type='gauss', sigma=1)]))
    assert not constant.varies()
    assert constant.at_velocities([0, 1]) == [constant, constant]


def test_varying_psfs_round_trip():
    info = dict(
        type='gauss', ratio=0.5,
        sigma=dict(wavelength=dict(values=[5000.0, 6000.0], unit='Angstrom'),
                   values=[1.0, 2.0]),
        posa=dict(velocity=dict(values=[0.0, 10.0], unit='km / s'),
                  values=dict(values=[0.1, 0.2], unit='rad')))
    psf = psf_parser.load(dict(info))
    dumped = psf_parser.dump(psf)
    assert psf_parser.dump(psf_parser.load(dict(dumped))) == dumped
    # (an angle in radians is converted to the degrees of posa)
    np.testing.assert_allclose(
        psf.at_velocities([10], 6000 * u.AA)[0].dump()['posa'],
        np.degrees(0.2))


@pytest.mark.parametrize('info, message', [
    (dict(type='hanning', width=dict(velocity=[0], values=[1])),
     "option 'width' cannot vary"),
    (dict(type='gauss', sigma=dict(velocity=[0, 1], values=[1, -1])),
     "sigma: sigma must be greater than 0"),
    (dict(type='gauss', sigma=dict(wavelength=[1, 2], values=[1, 1])),
     "sigma.wavelength: the wavelength needs a unit"),
    (dict(type='gauss', sigma=dict(velocity=[0, 0], values=[1, 1])),
     "must be distinct")])
def test_varying_options_are_checked(info, message):
    with pytest.raises(ConfigError, match=message):
        lsf_parser.load(info)


def test_tables_in_wavelength_need_the_rest():
    psf = psf_parser.load(
        dict(type='gauss', sigma=alpha_table(None, 'inline')))
    with pytest.raises(ConfigError, match="needs the rest"):
        psf.at_velocities([0])


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
    for psf, image in zip(cube.at_velocities([0, 150, 300]), expected):
        np.testing.assert_allclose(
            psf.asarray((1, 1), (21, 21)),
            PSFImage(image).asarray((1, 1), (21, 21)), atol=1e-12)


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
    assert dumped['wavelength']['unit'] == 'm'
    np.testing.assert_allclose(
        dumped['wavelength']['values'], [6.5e-7, 6.6e-7])
    again = psf_parser.load(dict(dumped))
    rest = 6.5e-7 * u.m
    for a, b in zip(psf.at_velocities([0, 1e3, 1e4], rest),
                    again.at_velocities([0, 1e3, 1e4], rest)):
        np.testing.assert_allclose(
            a.asarray((0.05, 0.05)), b.asarray((0.05, 0.05)))


def test_lsf_images_from_files(tmp_path):
    # One profile per row: the points come from the spectral axis of the
    # rows, and the width of the channels from the columns (km/s)
    import astropy.io.fits
    import astropy.wcs
    from gbkfit.instrument import LSFGauss, LSFImage
    profiles = np.stack([
        LSFGauss(sigma).asarray(5, 41) for sigma in (20, 40)])
    wcs = astropy.wcs.WCS(naxis=2)
    wcs.wcs.ctype = ['', 'WAVE']
    wcs.wcs.cunit = ['km/s', 'Angstrom']
    wcs.wcs.cdelt = [5, 2000]
    wcs.wcs.crval = [0, 5000]
    wcs.wcs.crpix = [21, 1]
    path = tmp_path / 'lsf_images.fits'
    astropy.io.fits.writeto(path, profiles, wcs.to_header())
    lsf = lsf_parser.load(dict(type='images', file=str(path)))
    assert lsf.varies()
    dumped = lsf_parser.dump(lsf, prefix=str(tmp_path / 'd_'))
    assert dumped['step'] == 5
    wavelengths = dumped['wavelength']
    np.testing.assert_allclose(
        u.Quantity(wavelengths['values'], wavelengths['unit']).to_value(u.AA),
        [5000, 7000])
    # Halfway between the points (6000 Angstrom), half of each profile
    rest = 6000 * u.AA
    middle, = lsf_parser.load(dict(dumped)).at_velocities([0], rest)
    np.testing.assert_allclose(
        middle.asarray(5, 41),
        LSFImage((profiles[0] + profiles[1]) / 2, 5).asarray(5, 41),
        atol=1e-12)
