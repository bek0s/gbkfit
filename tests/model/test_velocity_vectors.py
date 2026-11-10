"""
Tests for the velocity of the components as vectors (the extra outputs
vxdata, vydata and vzdata, along the axes of the grid): their part along
the line of sight is their velocity there (vdata) without the systemic
velocity, and their length the speed of their traits, for each kind of
component that moves.
"""

import numpy as np
import pytest


LINES = dict(type='lines')
UNIFORM = dict(type='uniform')
THICK = dict(
    rptraits=dict(type='exponential'), rhtraits=dict(type='sech2', zmax=4))
GEOMETRY = dict(xpos=0.5, ypos=-0.5, posa=30, incl=55, vsys=20)

# A speed along each direction of a disk (but outward), and a flow along a
# bar, in the plane
SPEEDS = [
    dict(type='uniform', direction='tan'),
    dict(type='uniform', direction='rad'),
    dict(type='uniform', direction='ver'),
    dict(type='bar')]
SPEED_VALUES = dict(
    vpt_vt=100, vpt2_vr=20, vpt3_vv=10, vpt4_v=30, vpt4_p=20, vpt4_s=2)


def disk(density, **options):
    """A disk of the given density, with the speeds of SPEEDS."""
    return dict(
        type='thick_disk' if 'rhtraits' in density else 'thin_disk',
        loose=False, tilted=False, rnodes=list(range(0, 9)),
        **density, emission=LINES, vptraits=SPEEDS, dptraits=UNIFORM) | options


# The components that move, and the values of their parameters
COMPONENTS = dict(
    thin_disk=(
        disk(dict(rptraits=dict(type='exponential'))),
        dict(rpt_a=1, rpt_s=3, dpt_a=10) | SPEED_VALUES),
    thick_disk=(
        disk(THICK), dict(rpt_a=1, rpt_s=3, rht_s=1, dpt_a=10) | SPEED_VALUES),
    cloud_disk=(
        disk(THICK, clouds=dict(cflux=1e-3)),
        dict(rpt_a=1, rpt_s=3, rht_s=1, dpt_a=10) | SPEED_VALUES),
    cone=(
        dict(
            type='cone', rnodes=[0, 1, 6], emission=LINES,
            rptraits=UNIFORM, rhtraits=dict(type='gauss', zmax=1.5),
            vptraits=[
                dict(type='uniform', direction='tan'),
                dict(type='uniform', direction='out')],
            dptraits=UNIFORM),
        dict(wall=30, rpt_a=1, rht_s=0.5, vpt_vt=100, vpt2_vo=50, dpt_a=10)),
    spheroid=(
        dict(
            type='spheroid', profile=dict(type='sersic'), rmax=6,
            emission=dict(type='band'),
            vptraits=dict(type='uniform', direction='tan'), dptraits=UNIFORM,
            rnodes=[0, 6]),
        dict(ie=1, re=2, n=1, vpt_vt=100, dpt_a=10)))


def case(driver, component):
    """
    A cube of a galaxy of the component (an image of a spheroid, which
    emits in bands), with a depth axis.
    """
    grid = dict(size=[24, 24], step=[0.5, 0.5])
    observable = dict(type='pixel_brightness') | grid \
        if component['type'] == 'spheroid' else dict(
            type='pixel_spectra', spectral=dict(
                size=41, step=10, convention='relativistic')) | grid
    return dict(
        driver=dict(type=driver.type()), observable=observable,
        line_of_sight=dict(type='grid', depth=dict(step=0.5)),
        galaxy=dict(components=[component]))


def vectors(extra):
    """Return the velocity as a vector of the component of the extras."""
    return np.stack([
        extra.get(f'v{axis}data', component='component0').data
        for axis in 'xyz'])


@pytest.mark.parametrize('name', COMPONENTS)
def test_the_vector_along_the_line_of_sight_is_the_velocity(
        driver, evaluate_cases, name):
    component, values = COMPONENTS[name]
    _, extra = evaluate_cases([case(driver, component)], GEOMETRY | values)
    vdata = extra.get('vdata', component='component0').data
    vector = vectors(extra)
    moves = np.isfinite(vdata)
    assert moves.sum() > 100
    assert np.isfinite(vector[:, moves]).all()
    np.testing.assert_allclose(
        vector[2][moves] + GEOMETRY['vsys'], vdata[moves], atol=1e-2)


def test_the_length_of_the_vector_is_the_speed(driver, evaluate_cases):
    # A smooth thick disk: one point of it in each voxel, of the speeds 100,
    # 20 and 10 km/s along the tangential, radial and vertical directions
    # (wherever the disk has them: off its axis and its midplane)
    component = disk(THICK) | dict(vptraits=SPEEDS[:3], dptraits=UNIFORM)
    _, extra = evaluate_cases(
        [case(driver, component)],
        GEOMETRY | dict(rpt_a=1, rpt_s=3, rht_s=1, dpt_a=10, vpt_vt=100,
                        vpt2_vr=20, vpt3_vv=10))
    length = np.linalg.norm(vectors(extra), axis=0)
    moves = np.isfinite(length)
    assert moves.sum() > 100
    np.testing.assert_allclose(
        np.median(length[moves]), np.sqrt(100 ** 2 + 20 ** 2 + 10 ** 2),
        rtol=1e-4)


def test_a_flow_along_a_bar_is_in_the_plane(driver, evaluate_cases):
    # A bar as wide as the disk: the flow of 30 km/s along its axis
    component = disk(THICK) | dict(vptraits=SPEEDS[3], dptraits=UNIFORM)
    _, extra = evaluate_cases(
        [case(driver, component)],
        GEOMETRY | dict(rpt_a=1, rpt_s=3, rht_s=1, dpt_a=10, vpt_v=30,
                        vpt_p=20, vpt_s=100))
    length = np.linalg.norm(vectors(extra), axis=0)
    moves = np.isfinite(length)
    assert np.mean(np.isclose(length[moves], 30, rtol=1e-4)) > 0.99


def test_the_total_vector_of_one_component_is_its_own(
        driver, evaluate_cases):
    component, values = COMPONENTS['thick_disk']
    _, extra = evaluate_cases([case(driver, component)], GEOMETRY | values)
    for axis in 'xyz':
        own = extra.get(f'v{axis}data', component='component0').data
        total = extra.get(f'total_v{axis}data').data
        np.testing.assert_allclose(total, own, equal_nan=True, rtol=1e-6)


def test_a_radial_flow_points_away_from_the_centre(driver, evaluate_cases):
    # The velocity of a radial flow of a disk is along the offset from its
    # centre, which the sky sees projected as the vector: (vx, vy) along
    # the offset on the sky, outwards (audit 2026-10-31 #86)
    component = dict(
        type='thin_disk', loose=False, tilted=False, rnodes=list(range(0, 9)),
        rptraits=dict(type='exponential'), emission=LINES,
        vptraits=dict(type='uniform', direction='rad'),
        dptraits=UNIFORM)
    _, extra = evaluate_cases([dict(
        driver=dict(type=driver.type()), observable=dict(
            type='pixel_spectra', size=[24, 24], step=[0.5, 0.5],
            spectral=dict(size=41, step=10, convention='relativistic')),
        line_of_sight=dict(type='flat'),
        galaxy=dict(components=[component]))],
        GEOMETRY | dict(rpt_a=1, rpt_s=3, vpt_vr=20, dpt_a=10))
    vx, vy = vectors(extra)[:2]
    y, x = (np.indices(vx.shape) - 11.5) * 0.5
    dx, dy = x - GEOMETRY['xpos'], y - GEOMETRY['ypos']
    moving = np.isfinite(vx) & (np.hypot(dx, dy) > 1)
    assert np.count_nonzero(moving) > 100
    cross = vx * dy - vy * dx
    np.testing.assert_allclose(
        cross[moving] / np.hypot(vx, vy)[moving] / np.hypot(dx, dy)[moving],
        0, atol=1e-6)
    assert np.all((vx * dx + vy * dy)[moving] > 0)
