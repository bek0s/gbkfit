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
     "the wavelength of a table needs a unit"),
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
