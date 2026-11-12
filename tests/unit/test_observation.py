"""
Tests for the making of observations: their observables from data, the
checks of their options, and the likelihoods of the observations of a
group (see Likelihoods).
"""

import astropy.units as u
import numpy as np
import pytest

from gbkfit.config import ConfigError, config_path
from gbkfit.dataset import Data, DatasetPixelMoments
from gbkfit.grid import SpectralAxis
from gbkfit.instrument import Instrument, LSFGauss
from gbkfit.observation import (
    FittedNoise, LikelihoodGauss, Likelihoods, LineOfSightFlat,
    ObservationGroup, ObservationSky, PixelBrightness, PixelMoments,
    UpperLimit, observable_parser, observation_parser)


def test_observable_from_data():
    # The data give the observable its grid and the orders of its
    # moments, and its spectral axis covers their velocities
    data = DatasetPixelMoments(
        {0: Data(np.ones((8, 20))), 1: Data(np.full((8, 20), 1500.0))},
        step=2, rota=30)
    observable = PixelMoments.from_data(
        data, spectral=SpectralAxis(step=5, convention='relativistic'))
    assert observable.grid() == data.grid()
    assert observable.orders() == (0, 1)
    assert observable.spectral().rval() == 1500
    ObservationSky(observable, LineOfSightFlat(), data=data)


@pytest.mark.parametrize('dataset_class, observable_class, shape', [
    ('DatasetPixelSpectra', 'PixelSpectra', (10, 4, 4)),
    ('DatasetSlitSpectra', 'SlitSpectra', (10, 4))])
def test_spectra_keep_the_convention_of_their_data(
        dataset_class, observable_class, shape):
    # Spectra on an axis of log velocities (WAVE-LOG) make an observable of
    # their grid, its convention included
    import gbkfit.dataset
    import gbkfit.observation
    data = getattr(gbkfit.dataset, dataset_class)(
        Data(np.zeros(shape)),
        spectral=SpectralAxis(convention='log', rest='6564.61 Angstrom'))
    observable = getattr(gbkfit.observation, observable_class).from_data(data)
    assert observable.grid() == data.grid()
    assert observable.grid().coords.convention == 'log'


def moments_group(names=('maps',), data=True):
    """
    A group of a disk and observations of moment maps of it of the given
    names (with or without data).
    """
    from gbkfit.model import galaxy_parser
    galaxy = galaxy_parser.load(dict(components=[dict(
        type='thin_disk', rnodes=[0, 1, 2], rptraits=dict(type='exponential'),
        vptraits=dict(type='arctan', direction='tan'),
        dptraits=dict(type='uniform'), emission=dict(type='lines'))]))
    dataset = DatasetPixelMoments(
        {0: Data(np.ones((8, 20))), 1: Data(np.full((8, 20), 1500.0))})
    observable = PixelMoments.from_data(
        dataset, spectral=SpectralAxis(step=5, convention='relativistic'))
    return ObservationGroup(galaxy, [
        ObservationSky(
            observable, LineOfSightFlat(), name=name,
            data=dataset if data else None)
        for name in names])


def test_a_model_that_can_have_missing_values_needs_their_residual():
    # A velocity map: the model has no velocity where its flux is below
    # the mask cutoff, and the residual there is the user's choice; the
    # default likelihood has none
    group = moments_group()
    with pytest.raises(ValueError, match="option missing_residual"):
        Likelihoods(group)
    Likelihoods(group, LikelihoodGauss(missing_residual=3))
    # (without data, only a likelihood given is checked: e.g. that of the
    # fit that a setup of mock data prepares)
    group = moments_group(data=False)
    assert Likelihoods(group).observations() == ()
    with pytest.raises(ValueError, match="option missing_residual"):
        Likelihoods(group, LikelihoodGauss())


def test_observable_options_are_checked():
    with pytest.raises(ValueError, match="'flux_floor' must be at least 0 and less than 1"):
        PixelMoments(
            size=(8, 8), flux_floor=1,
            spectral=SpectralAxis(convention='relativistic'))
    with pytest.raises(ValueError, match="no spectral axis for an LSF"):
        ObservationSky(
            PixelBrightness(size=(8, 8)), LineOfSightFlat(),
            instrument=Instrument(lsf=LSFGauss(10 * u.km / u.s)))


def test_grid_options_take_one_value_or_one_for_each_axis():
    # (audit 2026-10-31 #52) In Python as in configurations: a value for
    # both axes on the sky, or one for each; more (e.g. for the channels
    # of a cube) are an error that names the option
    observable = PixelBrightness(size=8, step=0.5)
    assert observable.size() == (8, 8)
    assert observable.step() == (0.5, 0.5)
    with pytest.raises(ValueError, match="option 'size' must have one"):
        PixelBrightness(size=(8, 8, 10))
    with pytest.raises(ValueError, match="option 'size' must have one"):
        observable_parser.load(dict(type='pixel_brightness', size=[8, 8, 10]))


def test_observations_need_their_type():
    info = dict(
        line_of_sight=dict(type='flat'),
        observable=dict(type='pixel_brightness', size=[4, 4]))
    with pytest.raises(ConfigError, match="missing required options: 'type'.*'sky'"):
        observation_parser.load(info)
    observation = observation_parser.load(info | dict(type='sky'))
    assert observation_parser.dump(observation)['type'] == 'sky'


def test_instrument_warnings_have_the_path_of_the_configuration(caplog):
    # The instrument is checked when the observation is made, so its
    # warnings have the path of the observation
    with config_path('observations'):
        observation_parser.load([dict(
            type='sky',
            line_of_sight=dict(type='flat'),
            observable=dict(
                type='pixel_spectra', size=[4, 4],
                spectral=dict(size=4, convention='relativistic'),
                smooth_weights=True))])
    assert "observations[0] [sky]: smooth_weights is true" in caplog.text


@pytest.mark.parametrize('dtype', ['float16', 'int32', 'complex'])
def test_dtypes_without_kernels_are_errors(dtype):
    # (they failed later, in the native code)
    dataset = Data(np.ones((8, 8)))
    from gbkfit.dataset import DatasetPixelBrightness
    observable = PixelBrightness.from_data(DatasetPixelBrightness(dataset))
    with pytest.raises(ValueError, match="'dtype' must be one of 'float32', 'float64'"):
        ObservationSky(observable, LineOfSightFlat(), dtype=dtype)


def test_ffts_are_float64_by_default_and_not_less_precise_than_models():
    observable = PixelBrightness(size=(8, 8))
    assert ObservationSky(
        observable, LineOfSightFlat()).fft_dtype() == np.float64
    with pytest.raises(ValueError, match=r"fft_dtype \(float32\) is less"):
        ObservationSky(
            observable, LineOfSightFlat(
                ), dtype='float64', fft_dtype='float32')


def brightness(unit=None, spatial_correlation=None):
    """An image of 8 x 8 pixels of 1 and errors 1, in the given unit."""
    from gbkfit.dataset import DatasetPixelBrightness
    return DatasetPixelBrightness(Data(
        np.ones((8, 8)), error=np.ones((8, 8)), unit=unit,
        spatial_correlation=spatial_correlation))


def test_the_data_choose_the_dtype_unless_it_is_given():
    # float64 for data of correlated noise, float32 otherwise: a copy of
    # other data chooses again, and dumps keep only a given dtype (the
    # copy kept the first choice, and dumps fixed it: audit 2026-11-11)
    from gbkfit.instrument import PSFGauss
    correlated = brightness(spatial_correlation=PSFGauss(1.0))
    observable = PixelBrightness.from_data(correlated)
    chosen = ObservationSky(observable, LineOfSightFlat())
    assert chosen.dtype() == np.float32
    assert 'dtype' not in chosen.dump()
    assert chosen.with_data(correlated).dtype() == np.float64
    given = ObservationSky(observable, LineOfSightFlat(), dtype='float32')
    assert given.with_data(correlated).dtype() == np.float32
    assert given.dump()['dtype'] == 'float32'


def image_group(*names, data=(True,)):
    """
    A group of a disk and images of it of the given names (one without a
    name by default), each with data (in Jy) or not.
    """
    from gbkfit.model import galaxy_parser
    galaxy = galaxy_parser.load(dict(components=[dict(
        type='thin_disk', rnodes=[0, 1, 2], rptraits=dict(type='exponential'),
        emission=dict(type='band'))]))
    dataset = brightness('Jy')
    observable = PixelBrightness.from_data(dataset)
    names = names or (None,) * len(data)
    return ObservationGroup([galaxy], [
        ObservationSky(
            observable, LineOfSightFlat(), name=name,
            data=dataset if with_data else None)
        for name, with_data in zip(names, data, strict=True)])


def test_the_parameters_of_likelihoods_have_the_units_of_their_data():
    # The parameters of a fitted noise are in the units of the data (see
    # Likelihood.param_units): those of a group with data of another unit
    # are in it (it kept the old: audit 2026-11-11), and not those of the
    # group, whose models do not read them
    group = image_group('obs')
    likelihoods = Likelihoods(group, LikelihoodGauss(fitted_noise=dict(
        brightness=FittedNoise(jitter=True))))
    jitter = 'obs_noise_jitter_brightness'
    assert jitter not in group.pdescs()
    assert likelihoods.pdescs()[jitter].unit() == u.Jy
    other = likelihoods.with_group(group.with_data({0: brightness('mJy')}))
    assert other.pdescs()[jitter].unit() == u.mJy
    assert other.of(0).noise_params('brightness') == (None, jitter)


def test_a_group_with_other_data_is_a_copy_of_the_same_parameters():
    # (a copy shares what the group made of each observation, and its
    # parameters, which the data do not change; the observations it gives
    # no data keep theirs)
    group = image_group('a', 'b', data=(True, False))
    copied = group.with_data({0: brightness('mJy')})
    assert copied.seen_by(0) is group.seen_by(0)
    assert copied.pdescs() == group.pdescs()
    assert copied.observations()[1] is group.observations()[1]
    assert copied.observations()[0].data()['brightness'].unit() == u.mJy
    assert group.with_data({1: brightness()}).observed() == (0, 1)
    with pytest.raises(ValueError, match="no observations \\[2\\]"):
        group.with_data({2: brightness()})


def test_likelihoods_are_one_for_all_or_by_observation_name():
    # (one for every observation; by name, the Gaussian for those left
    # out; the parameters are named after their observations)
    fitted = LikelihoodGauss(
        weights=2.0, fitted_noise=dict(brightness=FittedNoise(jitter=True)))
    group = image_group('a', 'b', 'c', data=(True, True, False))
    one = Likelihoods(group, fitted)
    assert one.observations() == group.observed() == (0, 1)
    assert set(one.pdescs()) == {
        'a_noise_jitter_brightness', 'b_noise_jitter_brightness'}
    assert one.given() is fitted
    by_name = Likelihoods(group, dict(b=fitted))
    assert by_name.of(0).weight('brightness') == 1.0
    assert by_name.of(1).weight('brightness') == 2.0
    assert set(by_name.pdescs()) == {'b_noise_jitter_brightness'}
    with pytest.raises(KeyError):
        by_name.of(2)
    # (a likelihood of an observation without data is checked against
    # its items, for the fit it prepares, and compares nothing)
    assert Likelihoods(group, dict(c=fitted)).observations() == (0, 1)
    with pytest.raises(ValueError, match=r"does not have: \['spectra'\]"):
        Likelihoods(group, dict(c=LikelihoodGauss(weights=dict(spectra=2))))


@pytest.mark.parametrize('names, data, given, message', [
    (('a', 'b'), (True, True), dict(d=LikelihoodGauss()),
     r"names unknown observations \['d'\]; those with names are \['a', "
     r"'b'\]"),
    ((None, None), (True, True), dict(observation0=LikelihoodGauss()),
     r"observations \[0, 1\] have no name: name them, or give one value "
     r"for all"),
    ((None, None), (False, False), dict(observation0=LikelihoodGauss()),
     r"unknown observations \['observation0'\]; those with names are"),
    ((None,), (True,), LikelihoodGauss(fitted_noise=dict(
        brightness=FittedNoise(jitter=True))),
     "observation 0: a likelihood with parameters"),
    (('a',), (True,), LikelihoodGauss(upper_limits=dict(
        bright=UpperLimit(limit=0.1, error=0.1))),
     r"observation 'a': the options of the likelihood name data items"),
])
def test_likelihoods_are_checked_against_their_observations(
        names, data, given, message):
    # (by name, each observation with data needs a name: the label of one
    # without, its position, would change with the order)
    with pytest.raises(ValueError, match=message):
        Likelihoods(image_group(*names, data=data), given)


def test_the_errors_of_likelihoods_by_name_are_at_their_names():
    with pytest.raises(ValueError) as info, config_path('likelihoods'):
        Likelihoods(image_group('a', 'b', data=(True, True)), dict(
            b=LikelihoodGauss(weights=dict(moment1=2))))
    assert info.value.config_path == ['likelihoods', 'b']


def test_likelihoods_have_parameters_of_names_of_their_own():
    # (the parameters of a likelihood must not be those of the group: e.g.
    # of a component of the name of the observation)
    from gbkfit.model import galaxy_parser
    from gbkfit.observation import LikelihoodUser
    likelihood = LikelihoodUser(lambda *args: 0.0, 'log_likelihood', ['s'])
    assert set(Likelihoods(image_group('a'), likelihood).pdescs()) == {'a_s'}
    galaxy = galaxy_parser.load(dict(components=[dict(
        type='thin_disk', name='disk', rnodes=[0, 1, 2],
        rptraits=dict(type='exponential'), emission=dict(type='band'))]))
    dataset = brightness()
    group = ObservationGroup(galaxy, ObservationSky(
        PixelBrightness.from_data(dataset), LineOfSightFlat(), data=dataset,
        name='disk'))
    clash = LikelihoodUser(lambda *args: 0.0, 'log_likelihood', ['rpt_a'])
    with pytest.raises(ValueError, match=r"names of parameters of the group:"
                                         r" \['disk_rpt_a'\]"):
        Likelihoods(group, clash)


def test_options_by_observation_are_one_for_all_or_by_name():
    group = image_group('a', 'b', 'c', data=(True, False, False))
    assert group.per_observation(5, 'noise') == {0: 5, 1: 5, 2: 5}
    assert group.per_observation(
        dict(a=1, b=2), 'noise', [0, 1]) == {0: 1, 1: 2}
    assert group.per_observation(
        dict(b=2), 'scheme', [1], complete=False) == {1: 2}
    with pytest.raises(ValueError, match=r"these have none: \['c'\]"):
        group.per_observation(dict(a=1, b=2), 'noise')
    unnamed = image_group(data=(True, False))
    assert unnamed.per_observation(5, 'noise') == {0: 5, 1: 5}
    with pytest.raises(ValueError, match=r"observations \[0, 1\] have no"):
        unnamed.per_observation({}, 'noise')


@pytest.mark.parametrize('options, message', [
    (dict(weights=dict(moment2=2)), r"does not have: \['moment2'\]"),
    (dict(upper_limits=dict(momnet0=UpperLimit(limit=1, error=0.1))),
     r"does not have: \['momnet0'\]")])
def test_likelihoods_name_items_of_their_observation(options, message):
    # A typo in the name of an item of the options of a likelihood would
    # leave the option out without a word
    with pytest.raises(ValueError, match=message):
        Likelihoods(moments_group(), LikelihoodGauss(
            missing_residual=3, **options))


def test_residuals_of_upper_limits_are_not_linear_in_the_model():
    # (so that no elements of the model are solved from them)
    from gbkfit.dataset import DatasetPixelSpectra
    from gbkfit.driver.drivers.host import DriverHost
    from gbkfit.objective import comparison_of
    from gbkfit.observation import PixelSpectra
    data = DatasetPixelSpectra(
        Data(np.ones((10, 4, 4))),
        spectral=SpectralAxis(convention='relativistic'))
    observation = ObservationSky(
        PixelSpectra.from_data(data), LineOfSightFlat(), data=data)
    comparison_ = comparison_of(observation, LikelihoodGauss(
        upper_limits=dict(spectra=UpperLimit(limit=0.1, error=0.05))),
        DriverHost())
    for form in (comparison_.linear_form, comparison_.pointwise_form):
        assert "has an upper limit" in form(['spectra'])


def test_the_psf_must_be_the_beam_of_data_of_a_beam(caplog):
    # Data of a beam (of a radio interferometer) are smoothed by it, a
    # Gaussian: a PSF of another shape or size is warned of; a beam for
    # each channel of a cube is that of the PSF at each, or one of them
    from gbkfit.dataset import DatasetPixelSpectra
    from gbkfit.instrument import PSFGauss, PSFMoffat, SpectralTable
    from gbkfit.io import Beam
    from gbkfit.observation import PixelSpectra
    hi = 1420.405752 * u.MHz
    spectral = SpectralAxis(size=3, step=10, rval=0, rest=hi)

    def warnings(psf, beam):
        caplog.clear()
        data = DatasetPixelSpectra(
            Data(np.zeros((3, 4, 4)), unit='Jy/beam', beam=beam),
            spectral=SpectralAxis(step=10, rval=0, rest=hi))
        ObservationSky(
            PixelSpectra(size=(4, 4), spectral=spectral), LineOfSightFlat(),
            instrument=Instrument(psf=psf), data=data)
        return caplog.text
    beam = Beam(6.0, 3.0, 30.0)
    assert warnings(PSFGauss.from_beam(6.0, 3.0, 30.0), beam) == ""
    assert warnings(PSFGauss.from_fwhm(6.0, 0.5, 210.0), beam) == ""
    assert "the PSF of the instrument is the beam 6 x 3 arcsec at 40" \
        in warnings(PSFGauss.from_beam(6.0, 3.0, 40.0), beam)
    assert "of type 'moffat', not a Gaussian" in warnings(
        PSFMoffat.from_fwhm(6.0, 2.5, 0.5, 30.0), beam)
    assert "is none (a point)" in warnings(None, beam)
    # (of a cube of CASA: a beam at the frequency of each channel)
    frequencies = [1420.4, 1420.36, 1420.31] * u.MHz
    beams = Beam(
        np.array([6.0, 6.1, 6.2]), np.full(3, 3.0), np.full(3, 30.0),
        frequencies)
    varying = PSFGauss.varying(
        bmaj=SpectralTable(frequencies, [6.0, 6.1, 6.2]), bmin=3.0,
        bpa=30.0)
    assert warnings(varying, beams) == ""
    text = warnings(PSFGauss.from_beam(6.0, 3.0, 30.0), beams)
    assert "another at 2 of their 3 channels: e.g. at 1420.36 MHz" in text
    assert "have one beam, 6 x 3 arcsec at 30 deg, but the PSF of the " \
        "instrument varies" in warnings(varying, beam)
