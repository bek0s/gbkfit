"""
Tests of the amplitudes of the light of components solved by the
objective (see Component.light_amplitudes and Objective): together, as
the least squares of the parts of the model (one for each amplitude), in
one or several observations, and refused where the model is not linear
in them.
"""

import copy

import numpy as np
import pytest
import scipy.optimize

from gbkfit.dataset import Data, DatasetPixelBrightness
from gbkfit.model import galaxy_parser
from gbkfit.objective import Objective
from gbkfit.observation import (
    LikelihoodGaussian, ObservationGroup, ObservationSky, PixelBrightness,
    line_of_sight_parser)
from gbkfit.params import ParamSpace
from gbkfit.utils.parseutils import ConfigError


# A core of two Gaussians (an MGE) and a Sersic envelope, in a band
CORE = dict(
    type='spheroid', name='core', rmax=20, emission=dict(type='band'),
    profile=dict(type='mge', ngauss=2))
ENVELOPE = dict(
    type='spheroid', name='envelope', rmax=20, emission=dict(type='band'),
    profile=dict(type='sersic'))
TRUTH = dict(
    xpos=0.2, ypos=-0.3, posa=30, incl=50, core_i0=[4.0, 1.5],
    core_sigma=[0.6, 1.8], core_q=[0.8, 0.6], envelope_ie=0.3,
    envelope_re=4.0, envelope_n=1.5)
AMPLITUDES = (('core_i0', 0), ('core_i0', 1), ('envelope_ie', None))
ERROR = 0.05

# Each amplitude solved above 0 (profiled: their integral has no exact
# formula), of a broad prior
SOLVE = dict(solve=dict(
    type='gaussian', mean=0, sigma=100, minimum=0, likelihood='profiled'))


def observation(size, data=None, observable=None):
    """An image of size pixels of 0.5 arcsec of the given data, if any."""
    dataset = None if data is None else DatasetPixelBrightness(
        Data(data, error=np.full_like(data, ERROR)), step=(0.5, 0.5))
    observable = observable or (
        PixelBrightness.from_data(dataset) if dataset is not None
        else PixelBrightness(size=size, step=(0.5, 0.5)))
    return ObservationSky(
        observable, line_of_sight_parser.load(dict(type='flat')),
        data=dataset)


def models(driver, sizes, values, galaxy=(CORE, ENVELOPE)):
    """The images of the galaxy of the given sizes, at the values."""
    group = ObservationGroup(
        [galaxy_parser.load(dict(components=copy.deepcopy(list(galaxy))))],
        [observation(size) for size in sizes])
    return [m['brightness']['d'].astype(float)
            for m in group.plan(driver).model_h(values)]


def with_amplitudes(values, amplitudes):
    """The values with the amplitudes (see AMPLITUDES) at the given ones."""
    values = copy.deepcopy(values)
    values['core_i0'] = list(amplitudes[:2])
    values['envelope_ie'] = float(amplitudes[2])
    return values


def test_solved_amplitudes_are_the_least_squares_of_their_parts(driver):
    # Two images (e.g. of two instruments) of the core and the envelope:
    # the three amplitudes of the light, of two components, fall on the
    # same pixels of both images, so they are solved together, from both:
    # those of the bounded least squares of the parts of the model (the
    # image of each amplitude alone), of the data and the prior
    sizes = ((32, 32), (24, 28))
    rng = np.random.default_rng(3)
    data = [m + ERROR * rng.normal(size=m.shape)
            for m in models(driver, sizes, TRUTH)]
    group = ObservationGroup(
        [galaxy_parser.load(dict(components=copy.deepcopy([CORE, ENVELOPE])))],
        [observation(None, d) for d in data])
    params = ParamSpace(
        group.pdescs(), TRUTH | dict(core_i0=SOLVE, envelope_ie=SOLVE))
    objective = Objective(group.plan(driver), params.solves())
    chi2 = objective.chi2(params.evaluate())
    parts = np.array([
        np.concatenate([m.ravel() for m in models(
            driver, sizes, with_amplitudes(TRUTH, np.eye(3)[k]))])
        for k in range(3)])
    rows = np.vstack((parts.T / ERROR, np.eye(3) / 100))
    target = np.concatenate((
        np.concatenate([d.ravel() for d in data]) / ERROR, np.zeros(3)))
    expected = scipy.optimize.lsq_linear(rows, target, bounds=(0, np.inf))
    solved = objective.solved()
    np.testing.assert_allclose(
        [solved['core_i0[0]'], solved['core_i0[1]'], solved['envelope_ie']],
        expected.x, rtol=1e-4)
    np.testing.assert_allclose(expected.x, [4.0, 1.5, 0.3], rtol=0.05)
    assert chi2 == pytest.approx(2 * expected.cost, rel=1e-4)
    assert 0 < objective.nnuisance() <= 3


@pytest.mark.parametrize('change, message', [
    # (the Jeans kinematics take the light: the stars move with it)
    (dict(kinematics=dict(type='jeans')), "jeans kinematics take its light"),
    # (a mass follows the light: it moves the gravity)
    (dict(mass=True), "a mass follows the light"),
    # (moments are ratios of sums of the light)
    (dict(moments=True), "not linear in the light"),
])
def test_amplitudes_of_models_not_linear_in_them_are_refused(
        driver, change, message):
    core = CORE | dict(emission=dict(type='stars'))
    if 'kinematics' in change:
        core |= dict(kinematics=change['kinematics'])
    galaxy = dict(components=[core])
    if 'kinematics' in change or 'mass' in change:
        galaxy |= dict(
            mass=[dict(type='light_spheroid', name='stars', component='core')]
            if 'mass' in change else [dict(type='point', name='bh')],
            distance=dict(type='direct', cosmology='Planck18'))
    data = np.zeros((16, 16))
    observable = None
    if 'moments' in change:
        from gbkfit.dataset import DatasetPixelMoments
        from gbkfit.observation import PixelMoments
        core |= dict(kinematics=dict(
            type='traits', vptraits=dict(type='uniform', direction='tan'),
            dptraits=dict(type='uniform')))
        moments = DatasetPixelMoments({1: Data(data, error=data + 1)})
        observable = PixelMoments.from_data(
            moments, spec_convention='relativistic')
        obs = ObservationSky(
            observable, line_of_sight_parser.load(
                dict(type='grid', depth=dict(step=0.5))),
            data=moments, lines=['stars'],
            likelihood=LikelihoodGaussian(missing_residual=3))
    else:
        obs = observation(None, data)
    group = ObservationGroup([galaxy_parser.load(galaxy)], [obs])
    pdescs = group.pdescs()
    properties = {name: 1.0 for name in pdescs if not pdescs[name].size() > 1}
    properties |= {name: [1.0] * pdescs[name].size()
                   for name in pdescs if pdescs[name].size() > 1}
    properties['core_i0'] = SOLVE
    params = ParamSpace(pdescs, properties)
    with pytest.raises(ConfigError, match=message):
        Objective(group.plan(driver), params.solves())


def test_amplitudes_with_a_solved_flux_scale_are_refused(driver):
    # (the model is the scale times the amplitudes: not linear in both)
    data = np.zeros((16, 16))
    dataset = DatasetPixelBrightness(Data(data, error=data + 1))
    obs = ObservationSky(
        PixelBrightness.from_data(dataset),
        line_of_sight_parser.load(dict(type='flat')), data=dataset,
        flux_scale='one', name='obs')
    group = ObservationGroup(
        [galaxy_parser.load(dict(components=[copy.deepcopy(CORE)]))], [obs])
    params = ParamSpace(group.pdescs(), dict(
        xpos=0, ypos=0, posa=0, incl=0, core_sigma=[1, 2], core_q=[1, 1],
        core_i0=SOLVE, obs_flux_scale=dict(solve=dict(
            type='gaussian', mean=1, sigma=0.5, minimum=0))))
    with pytest.raises(ConfigError, match="not linear in both"):
        Objective(group.plan(driver), params.solves())
