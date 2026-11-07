import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from gbkfit.io import read_column, read_table


@pytest.fixture
def velocities(tmp_path):
    path = tmp_path / 'velocities.ecsv'
    Table({'v': [1.0, 2.0] * u.km / u.s}).write(path)
    return read_table(str(path))


def test_a_column_is_read_in_its_own_unit(velocities):
    values = read_column(velocities, 'v', None)
    assert values.unit == u.km / u.s
    np.testing.assert_array_equal(values.value, [1, 2])


def test_a_column_with_a_unit_is_converted_to_the_unit_given(velocities):
    values = read_column(velocities, 'v', 'm/s')
    assert values.unit == u.m / u.s
    np.testing.assert_array_equal(values.value, [1000, 2000])


def test_a_column_is_not_converted_to_a_unit_of_another_kind(velocities):
    with pytest.raises(ValueError, match="cannot be converted to 'kpc'"):
        read_column(velocities, 'v', 'kpc')


def test_a_column_without_a_unit_is_in_the_unit_given(tmp_path):
    path = tmp_path / 'velocities.csv'
    path.write_text('v\n1\n2\n')
    table = read_table(str(path))
    values = read_column(table, 'v', 'km/s')
    assert values.unit == u.km / u.s
    np.testing.assert_array_equal(values.value, [1, 2])


def test_a_table_of_a_format_astropy_cannot_tell_needs_its_format(tmp_path):
    path = tmp_path / 'velocities.xyz'
    path.write_text('v\n1\n2\n')
    with pytest.raises(ValueError, match="give it"):
        read_table(str(path))
    table = read_table(str(path), 'ascii.csv')
    assert table.colnames == ['v']


def test_the_units_of_a_fits_table_are_read_by_the_fits_standard(tmp_path):
    # (astropy alone reads 'Jy/beam km/s' as Jy / (beam km s)); a unit
    # that is not one is an error
    path = str(tmp_path / 'table.fits')
    columns = [fits.Column('m0', 'D', unit='Jy/beam km/s', array=[1.0]),
               fits.Column('v', 'D', unit='km/s', array=[2.0])]
    fits.BinTableHDU.from_columns(columns).writeto(path)
    table = read_table(path)
    assert read_column(table, 'm0', None).unit == u.Jy * u.km / u.beam / u.s
    assert read_column(table, 'v', None).unit == u.km / u.s
    columns[1] = fits.Column('v', 'D', unit='KM/S', array=[2.0])
    fits.BinTableHDU.from_columns(columns).writeto(path, overwrite=True)
    with pytest.raises(ValueError, match="column 'v', 'KM/S', is not a"):
        read_table(path)


def test_the_hdu_of_a_fits_table_must_hold_a_table(tmp_path):
    # (astropy reads the first table in place of an HDU that holds none)
    path = str(tmp_path / 'table.fits')
    table = fits.BinTableHDU.from_columns(
        [fits.Column('v', 'D', unit='km/s', array=[2.0])], name='KIN')
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU([1.0]), table]).writeto(
        path)
    for hdu in (None, 2, 'KIN'):
        assert read_column(read_table(path, hdu=hdu), 'v', None).unit \
            == u.km / u.s
    for hdu in (0, 1, 5, 'NOPE'):
        with pytest.raises(ValueError, match="no table in HDU"):
            read_table(path, hdu=hdu)


def test_the_units_of_an_ecsv_file_are_read_by_the_fits_standard(tmp_path):
    # (astropy alone reads 'Jy/beam km/s' as Jy / (beam km s))
    path = tmp_path / 'table.ecsv'
    header = ('# %ECSV 1.0\n# ---\n# datatype:\n'
              '# - {name: m0, unit: Jy/beam km/s, datatype: float64}\n')
    path.write_text(header + 'm0\n1.0\n')
    table = read_table(str(path))
    assert read_column(table, 'm0', None).unit == u.Jy * u.km / u.beam / u.s
    path.write_text(header.replace('Jy/beam km/s', 'banana') + 'm0\n1.0\n')
    with pytest.raises(ValueError, match="column 'm0', 'banana', is not a"):
        read_table(str(path))


def test_the_format_of_a_table_must_be_one_astropy_reads(tmp_path):
    path = str(tmp_path / 'table.ecsv')
    Table(dict(v=[1.0])).write(path)
    with pytest.raises(ValueError, match="no tables of the format 'FITS'"):
        read_table(path, 'FITS')
    with pytest.raises(ValueError, match="table.ecsv: not a FITS file"):
        read_table(path, 'fits')
