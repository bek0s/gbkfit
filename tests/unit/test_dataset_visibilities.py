"""
Tests of the datasets of visibilities: their file, their checks, and the
reading of measurement sets.
"""

import numpy as np
import pytest

from gbkfit.dataset import (
    Data, DatasetVisibilities, UVCoverage, dataset_parser)
from gbkfit.utils.parseutils import ConfigError


HI = '1420.405752 MHz'
FREQUENCIES = 1.42e9 - np.arange(3) * 1e5


def visibilities(frequencies=FREQUENCIES, frame='LSRK'):
    rng = np.random.default_rng(1)
    shape = (len(frequencies), 4)
    error = rng.uniform(1, 2, shape)
    error[0, 1] = np.inf
    return DatasetVisibilities(
        Data(rng.standard_normal(shape), error=error, unit='Jy'),
        Data(rng.standard_normal(shape), error=error, unit='Jy'),
        UVCoverage(
            rng.uniform(-1e3, 1e3, (4, 3)), frequencies, (150, -30), frame,
            HI))


def test_the_file_of_a_dataset_is_the_dataset():
    dataset = visibilities()
    loaded = dataset_parser.load(dataset_parser.dump(dataset))
    for key in ('re', 'im'):
        np.testing.assert_array_equal(loaded[key].data(), dataset[key].data())
        np.testing.assert_array_equal(
            loaded[key].error(), dataset[key].error())
        assert loaded[key].unit() == dataset[key].unit()
    assert np.isnan(loaded['re'].data()[0, 1])
    assert loaded.coverage() == dataset.coverage()


@pytest.mark.parametrize('options, message', [
    (dict(frame='TOPO'), "regrid"),
    (dict(frequencies=[1.42e9, 1.4201e9, 1.4203e9]), "uniform")])
def test_the_channels_are_checked(options, message):
    with pytest.raises(ConfigError, match=message):
        visibilities(**options)


def write_ms(path, uvw, data, flags, weights, frequencies, rest=None):
    # A measurement set of one field and one spectral window (LSRK), with
    # the cross-correlations XX and YY of the baselines
    tables = pytest.importorskip('casacore.tables')
    nrows = len(uvw)
    with tables.default_ms(path, tables.maketabdesc(
            [tables.makearrcoldesc('DATA', 0j, ndim=2)])) as ms:
        ms.addrows(nrows)
        ms.putcol('UVW', uvw)
        ms.putcol('DATA', data)
        ms.putcol('FLAG', flags)
        ms.putcol('WEIGHT', weights)
        ms.putcol('SIGMA', 1 / np.sqrt(weights))
        ms.putcol('ANTENNA1', np.zeros(nrows, int))
        ms.putcol('ANTENNA2', np.arange(1, nrows + 1))
    rows = dict(
        FIELD=dict(PHASE_DIR=np.radians([[[150, -30]]]), NAME=['galaxy'],
                   SOURCE_ID=[0]),
        SPECTRAL_WINDOW=dict(CHAN_FREQ=[frequencies], MEAS_FREQ_REF=[1]),
        POLARIZATION=dict(CORR_TYPE=[[9, 12]], NUM_CORR=[2]),
        DATA_DESCRIPTION=dict(SPECTRAL_WINDOW_ID=[0], POLARIZATION_ID=[0]))
    for name, columns in rows.items():
        with tables.table(f'{path}/{name}', readonly=False, ack=False) as t:
            t.addrows(1)
            for column, value in columns.items():
                t.putcol(column, np.array(value))
    if rest is not None:
        with tables.default_ms_subtable(
                'SOURCE', f'{path}/SOURCE', tables.maketabdesc(
                    [tables.makearrcoldesc('REST_FREQUENCY', 0.0, ndim=1)])) \
                as t:
            t.addrows(1)
            t.putcol('SOURCE_ID', np.array([0]))
            t.putcell('REST_FREQUENCY', 0, np.array([rest]))
        with tables.table(path, readonly=False, ack=False) as ms:
            ms.putkeyword('SOURCE', f'Table: {path}/SOURCE')


def test_measurement_sets_are_read_as_stokes_i():
    # I = (XX + YY) / 2, of the error (sigma_XX^2 + sigma_YY^2)^(1/2) / 2,
    # left out where either is flagged
    rng = np.random.default_rng(2)
    uvw = rng.uniform(-1e3, 1e3, (5, 3))
    data = rng.standard_normal((5, 3, 2)) + 1j * rng.standard_normal((5, 3, 2))
    flags = np.zeros((5, 3, 2), bool)
    flags[1, 2, 1] = True
    weights = rng.uniform(1, 4, (5, 2))
    write_ms('test.ms', uvw, data, flags, weights, FREQUENCIES, 1.4204e9)
    dataset = DatasetVisibilities.from_ms('test.ms')
    stokes = (data[..., 0] + data[..., 1]).T / 2
    for key, part in dict(re=stokes.real, im=stokes.imag).items():
        part[2, 1] = np.nan
        np.testing.assert_allclose(dataset[key].data(), part)
    error = np.sqrt(1 / weights[:, 0] + 1 / weights[:, 1]) / 2
    np.testing.assert_allclose(dataset['re'].error()[0], error)
    coverage = dataset.coverage()
    np.testing.assert_allclose(coverage.uvw(), -uvw)
    np.testing.assert_allclose(coverage.phase_centre(), (150, -30))
    assert coverage.frame() == 'LSRK' and coverage.rest().value == 1.4204e9


def test_measurement_sets_need_a_rest_frequency():
    rng = np.random.default_rng(3)
    write_ms(
        'test.ms', rng.uniform(-1e3, 1e3, (2, 3)), np.ones((2, 3, 2), complex),
        np.zeros((2, 3, 2), bool), np.ones((2, 2)), FREQUENCIES)
    with pytest.raises(ConfigError, match="rest"):
        DatasetVisibilities.from_ms('test.ms')
    dataset = DatasetVisibilities.from_ms('test.ms', rest=HI)
    assert dataset.coverage().rest().unit == 'Hz'
