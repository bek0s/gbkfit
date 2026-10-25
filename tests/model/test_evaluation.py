"""
Tests for the evaluation contract of galaxies: evaluating a galaxy must
not modify its inputs, and must not depend on previous evaluations.
"""

import copy
import os
import pathlib
import subprocess
import sys

import gbkfit.params
import numpy as np
import pytest
from modelutils import group_plan


def line_disk(vptraits, **options):
    """
    A galaxy with a thin disk that emits lines, with the given velocity
    traits and options.
    """
    return dict(components=[dict(
        type='smdisk', loose=False, tilted=False,
        rnodes=list(range(0, 12)),
        density=dict(ptraits=dict(type='exponential')),
        kinematics=dict(vptraits=vptraits, dptraits=dict(type='uniform')),
        emission=dict(type='lines')) | options])


def nodewise_relative_case(driver):
    """
    A galaxy with a node-wise rotation curve, whose values are given as
    offsets (see test_evaluation_does_not_modify_params).
    """
    return dict(
        driver=dict(type=driver.type()),
        observable=dict(
            type='pixel_spectra', size=[32, 32, 41], step=[1, 1, 10]),
        line_of_sight=dict(type='flat'),
        galaxy=line_disk(dict(type='nw_uniform', direction='tan')))


def test_evaluation_does_not_modify_params(driver):
    plan = group_plan([nodewise_relative_case(driver)])
    params = gbkfit.params.ParamSpace(plan.group().pdescs(), dict(
        vsys=0, xpos=0, ypos=0, posa=30, incl=45,
        pt_a=1, pt_s=4, dpt_a=10,
        vpt_vt=[100] + [10] * 11),
        modes=dict(vpt_vt=gbkfit.params.ParamModeOffsets()))
    values = params.evaluate()
    values_before = copy.deepcopy(values)
    first = copy.deepcopy(plan.model_h(values)[0]['spectra']['d'])
    second = plan.model_h(values)[0]['spectra']['d']
    np.testing.assert_array_equal(values['vpt_vt'], values_before['vpt_vt'])
    np.testing.assert_array_equal(first, second)


# Evaluates a thick smooth disk on the host and saves its spectral cube.
# Argument: the output file.
_THICK_DISK_SCRIPT = """
import sys
import gbkfit.params
import numpy as np
from modelutils import group_plan
plan = group_plan([dict(
    driver=dict(type='host'),
    observable=dict(type='pixel_spectra', size=[24, 24, 31], step=[1, 1, 10]),
    line_of_sight=dict(type='grid'),
    galaxy=dict(components=[dict(
        type='smdisk', loose=False, tilted=False,
        rnodes=list(range(0, 12)),
        density=dict(
            ptraits=dict(type='exponential'),
            htraits=dict(type='sech2', zmax=8)),
        kinematics=dict(
            vptraits=dict(type='arctan', direction='tan'), dptraits=dict(type='uniform')),
        emission=dict(type='lines'))]))])
params = gbkfit.params.ParamSpace(plan.group().pdescs(), dict(
    vsys=0, xpos=0, ypos=0, posa=30, incl=60, pt_a=1, pt_s=4, ht_s=1,
    vpt_rt=2, vpt_vt=150, dpt_a=20))
np.save(sys.argv[1], plan.model_h(params.evaluate())[0]['spectra']['d'])
"""


def test_host_smooth_disk_does_not_depend_on_thread_count(tmp_path):
    # Each thread adds the voxels of its spaxels in order, so a thick
    # disk is the same to the bit with any number of threads
    cubes = []
    for threads in (1, 7):
        output = tmp_path / f'threads_{threads}.npy'
        subprocess.run(
            [sys.executable, '-c', _THICK_DISK_SCRIPT, str(output)],
            env=os.environ | dict(OMP_NUM_THREADS=str(threads)),
            cwd=pathlib.Path(__file__).parent, check=True)
        cubes.append(np.load(output))
    assert cubes[0].any()
    np.testing.assert_array_equal(cubes[1], cubes[0])


# A galaxy seen through a flat line of sight by each kind of observation
# (a spectral cube and an image): the galaxy, its observable, the size
# and the data key of its observable, and its parameter properties
FLAT_OBSERVATIONS = dict(
    spectral_cube=(
        line_disk(dict(type='arctan', direction='tan')),
        'pixel_spectra', (32, 32, 41), 'spectra',
        dict(vsys=0, xpos=0, ypos=0, posa=30, incl=45,
             pt_a=1, pt_s=4, vpt_rt=2, vpt_vt=100, dpt_a=10)),
    image=(
        dict(components=[dict(
            type='smdisk', loose=False, tilted=False,
            rnodes=list(range(0, 12)),
            density=dict(ptraits=dict(type='exponential')),
            emission=dict(type='band'))]),
        'pixel_brightness', (32, 32), 'brightness',
        dict(xpos=0, ypos=0, posa=30, incl=45, pt_a=1, pt_s=4)))


@pytest.mark.parametrize('kind', FLAT_OBSERVATIONS)
def test_one_galaxy_observed_on_grids_with_different_steps(driver, kind):
    # Observations of one galaxy on grids with different steps each get
    # the galaxy on their own grid
    from gbkfit.model import galaxy_parser
    from gbkfit.observation import (
        Flat, ObservationGroup, ObservationSky, observable_parser)
    galaxy_info, observable_type, size, key, properties = \
        FLAT_OBSERVATIONS[kind]

    def observation(step):
        steps = (step, step, 10)[:len(size)]
        return ObservationSky(observable_parser.load(dict(
            type=observable_type, size=list(size), step=list(steps))),
            Flat())

    def evaluate(group, i):
        params = gbkfit.params.ParamSpace(group.pdescs(), properties)
        return group.plan(driver).model_h(
            params.evaluate())[i][key]['d'].copy()

    shared = ObservationGroup(
        [galaxy_parser.load(galaxy_info)], [observation(1), observation(0.5)])
    alone = ObservationGroup(
        [galaxy_parser.load(galaxy_info)], [observation(0.5)])
    np.testing.assert_array_equal(evaluate(shared, 1), evaluate(alone, 0))


@pytest.mark.parametrize('observable, emission', [
    (dict(type='pixel_brightness', size=[8, 8]), 'lines'),
    (dict(type='pixel_spectra', size=[8, 8, 8]), 'band')])
def test_incompatible_observations_are_rejected_before_evaluation(
        observable, emission):
    # An image sees the components that emit in a band, and the others
    # (spectral cubes) those that emit lines; the plan of the group of an
    # observation that sees none is rejected before any evaluation
    component = dict(
        type='smdisk', loose=False, tilted=False, rnodes=[0, 1, 2],
        density=dict(ptraits=dict(type='exponential')),
        emission=dict(type=emission))
    if emission == 'lines':
        component |= dict(kinematics=dict(
            vptraits=dict(type='arctan', direction='tan'), dptraits=dict(type='uniform')))
    case = dict(
        driver=dict(type='host'), observable=observable,
        line_of_sight=dict(type='flat'),
        galaxy=dict(components=[component]))
    with pytest.raises(Exception, match="it sees none"):
        group_plan([case])


def test_a_velocity_that_is_not_a_number_makes_spectra_nan(driver):
    # A NaN velocity (here of a rotation curve whose parameters are out of
    # their range: a negative power of a negative number within -rt) makes
    # the spectra NaN, so that it is seen, instead of picking channels
    # out of the cube
    plan = group_plan([dict(
        driver=dict(type=driver.type()),
        observable=dict(
            type='pixel_spectra', size=[16, 16, 21], step=[1, 1, 10]),
        line_of_sight=dict(type='flat'),
        galaxy=line_disk(
            dict(type='courteau', direction='tan'), rnodes=list(range(0, 8))))])
    params = gbkfit.params.ParamSpace(plan.group().pdescs(), dict(
        vsys=0, xpos=0, ypos=0, posa=30, incl=45, pt_a=1, pt_s=4,
        vpt_rt=-3, vpt_vt=100, vpt_b=0.4, vpt_g=2, dpt_a=10))
    cube = plan.model_h(params.evaluate())[0]['spectra']['d']
    nan = np.isnan(cube)
    assert nan.any() and np.isfinite(cube[~nan]).all()
    # The spectra of the spaxels within -rt are NaN
    assert (nan.all(axis=0) == nan.any(axis=0)).all()


def test_cuda_smooth_disk_of_more_voxels_than_an_int_holds(driver):
    # The voxels of a smooth disk need not be in memory: a grid with a
    # depth axis can have more than an int holds (here 64 x 64 x 528384),
    # which must give the image of the same depth in fewer voxels (it gave
    # zeros)
    if driver.type() != 'cuda':
        pytest.skip("the host takes minutes for so many voxels")

    def image(depth_size, depth_step):
        depth = dict(extent=depth_size * depth_step, step=depth_step)
        plan = group_plan([dict(
            driver=dict(type=driver.type()),
            observable=dict(type='pixel_brightness', size=[64, 64]),
            line_of_sight=dict(type='grid', depth=depth),
            galaxy=dict(components=[dict(
                type='smdisk', loose=False, tilted=False,
                rnodes=list(range(0, 30)),
                density=dict(
                    ptraits=dict(type='exponential'),
                    htraits=dict(type='sech2', zmax=8)),
                emission=dict(type='band'))]))])
        params = gbkfit.params.ParamSpace(plan.group().pdescs(), dict(
            xpos=0, ypos=0, posa=0, incl=30, pt_a=1, pt_s=6, ht_s=1))
        return plan.model_h(params.evaluate())[0]['brightness']['d']
    many = image(528384, 0.001)
    few = image(1032, 0.512)
    assert few.sum() > 0
    assert many.sum() == pytest.approx(few.sum(), rel=0.01)



def two_observations(driver):
    """
    A plan of two spectral cubes of a disk, 'a' and 'b' (which fits the
    offset of its frame).
    """
    from gbkfit.model import galaxy_parser
    from gbkfit.observation import (
        Flat, ObservationGroup, ObservationSky, PixelSpectra)
    return ObservationGroup(
        [galaxy_parser.load(line_disk(dict(type='arctan', direction='tan')))],
        [ObservationSky(
            PixelSpectra(size=(16, 16, 11), step=(1, 1, 20)), Flat(),
            name=name, offsets=dict(position=name == 'b'))
         for name in ('a', 'b')]).plan(driver)


def counted(plan):
    """
    Return the counts of the evaluations of the stage of the galaxy and
    of the models of the observations of a plan of two_observations.
    """
    counts = dict(galaxy=0, a=0, b=0)

    def counting(name, evaluate):
        def evaluated(*args):
            counts[name] += 1
            return evaluate(*args)
        return evaluated
    for name, part in zip(('galaxy', 'a', 'b'), plan._stages + plan._plans):
        part.evaluate = counting(name, part.evaluate)
    return counts


TWO_PARAMS = dict(
    vsys=0, xpos=0, ypos=0, posa=30, incl=45, pt_a=1, pt_s=4, vpt_rt=2,
    vpt_vt=150, dpt_a=20, b_dx=0, b_dy=0)


def spectra(models):
    """Copies of the spectra of the models of observations."""
    return [model['spectra']['d'].copy() for model in models]


def test_evaluations_skip_what_reads_no_changed_parameter(driver):
    # The offset of the frame of b moves b only: the galaxy and a are not
    # evaluated again, and the models are those of a new plan
    plan = two_observations(driver)
    counts = counted(plan)
    before = spectra(plan.model_h(TWO_PARAMS))
    moved = TWO_PARAMS | dict(b_dx=0.5)
    after = spectra(plan.model_h(moved))
    assert counts == dict(galaxy=1, a=1, b=2)
    for got, want in zip(
            after, spectra(two_observations(driver).model_h(moved))):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(after[0], before[0])
    assert not np.array_equal(after[1], before[1])
    # (the disk moves them all; extra outputs evaluate all, as NaN does)
    plan.model_h(moved | dict(pt_s=3))
    assert counts == dict(galaxy=2, a=2, b=3)
    from gbkfit.model import Extras
    plan.model_h(moved | dict(pt_s=3), Extras())
    assert counts == dict(galaxy=3, a=3, b=4)
    plan.model_h(moved | dict(pt_s=np.nan))
    plan.model_h(moved | dict(pt_s=np.nan))
    assert counts == dict(galaxy=5, a=5, b=6)


def test_mocks_and_plans_with_other_data_evaluate_again(driver):
    # Mocks write their draws over the models; plans with other data share
    # the models (of data of the same unit), and the last evaluations
    from gbkfit.observation import NoiseGauss
    noise = NoiseGauss(rms=1.0)
    first = two_observations(driver).with_data([
        mock for mock, in two_observations(driver).mock(
            TWO_PARAMS, noise, seed=1)])
    counts = counted(first)
    models = spectra(first.model_h(TWO_PARAMS))
    mocks = first.mock(TWO_PARAMS, noise, seed=2)
    for got, want in zip(spectra(first.model_h(TWO_PARAMS)), models):
        np.testing.assert_array_equal(got, want)
    assert counts == dict(galaxy=1, a=2, b=2)
    second = first.with_data([mock for mock, in mocks])
    assert second._plans == first._plans
    second.model_h(TWO_PARAMS | dict(vpt_vt=100))
    for got, want in zip(spectra(first.model_h(TWO_PARAMS)), models):
        np.testing.assert_array_equal(got, want)
    assert counts == dict(galaxy=3, a=4, b=4)


def test_observations_on_two_drivers_evaluate_at_once(driver):
    # Each in a thread of its driver, as on one driver: a on the host, b
    # on the driver (or on another host driver), with the extra outputs
    # in the order of one driver
    import threading
    from gbkfit.driver.drivers.host import DriverHost
    from gbkfit.model import Extras
    alone = {
        'host': two_observations(DriverHost()),
        'driver': two_observations(driver)}
    split = alone['host'].group().plan(
        DriverHost(), dict(b=driver if driver.type() != 'host'
                           else DriverHost()))
    threads = []

    def recording(evaluate):
        def evaluated(*args):
            threads.append(threading.current_thread().name)
            return evaluate(*args)
        return evaluated
    for part in split._plans:
        part.evaluate = recording(part.evaluate)
    extras = dict(split=Extras(), alone=Extras())
    a, b = spectra(split.model_h(TWO_PARAMS, extras['split']))
    np.testing.assert_array_equal(
        a, spectra(alone['host'].model_h(TWO_PARAMS))[0])
    np.testing.assert_array_equal(
        b, spectra(alone['driver'].model_h(TWO_PARAMS, extras['alone']))[1])
    assert len(set(threads)) == 2
    assert all(name.startswith('gbkfit-driver') for name in threads)
    assert [(x.name, x.ids) for x in extras['split']] \
        == [(x.name, x.ids) for x in extras['alone']]


def test_an_error_on_one_of_two_drivers_is_raised(driver):
    from gbkfit.driver.drivers.host import DriverHost
    plan = two_observations(driver).group().plan(
        driver, dict(b=DriverHost()))
    counts = counted(plan)
    evaluate = plan._plans[1].evaluate

    def failing(*args):
        raise RuntimeError("b failed")
    plan._plans[1].evaluate = failing
    with pytest.raises(RuntimeError, match="b failed"):
        plan.model_h(TWO_PARAMS)
    assert counts['a'] == 1
    plan._plans[1].evaluate = evaluate
    plan.model_h(TWO_PARAMS)
    assert counts == dict(galaxy=1, a=1, b=1)
