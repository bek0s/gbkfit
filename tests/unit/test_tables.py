import astropy.units as u
import numpy as np
import pytest
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
