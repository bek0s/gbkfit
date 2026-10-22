"""
Tests of the Eryn fitter's layout: which rows of a table of blobs it
varies, and how its leaves make the free values.
"""

import numpy as np
import pytest

from gbkfit.model import galaxy_parser
from gbkfit.params import ParamSpace

pytest.importorskip('eryn')
from gbkfit.fitting.fitters.eryn.core import (  # noqa: E402
    FitParamEryn, _layout)


GALAXY = dict(components=[dict(
    type='smdisk', loose=False, tilted=False, rnodes=[0, 5, 10],
    density=dict(ptraits=[
        dict(type='exponential'), dict(type='mixture_gauss', nblobs=3)]),
    emission=dict(type='band'))])

PROPERTIES = dict(
    xpos=0, ypos=0, posa=30, incl=60, pt_a=dict(min=0, max=2), pt_s=2,
    pt1_r=dict(min=0, max=10), pt1_t=dict(min=0, max=360),
    pt1_a=dict(min=0, max=2), pt1_s=1, pt1_q=1, pt1_p=0)


def layout(properties):
    """The layout of the galaxy with the given properties."""
    pdescs = galaxy_parser.load(GALAXY).pdescs()
    params = ParamSpace(pdescs, properties)
    names = tuple(params.free_properties())
    fit_params = [FitParamEryn.load(info)
                  for info in params.free_properties().values()]
    return _layout(params, names, fit_params, {}), names


def test_rows_that_share_their_settings_are_varied():
    lay, names = layout(PROPERTIES)
    (table,) = lay.tables
    assert table.name == 'pt1' and table.columns == ('r', 't', 'a')
    assert [[names[i] for i in row] for row in table.elements] == [
        [f'pt1_{c}[{i}]' for c in 'rta'] for i in range(3)]
    assert [names[i] for i in lay.fixed] == ['pt_a']
    # Rows with settings of their own are always used
    own = {k: v for k, v in PROPERTIES.items() if k != 'pt1_r'} | {
        'pt1_r[0]': dict(min=2, max=3), 'pt1_r[1:]': dict(min=0, max=10)}
    lay, names = layout(own)
    assert not lay.tables
    assert 'pt1_r[0]' in [names[i] for i in lay.fixed]
    # Blobs of a fixed amplitude cannot be dropped
    assert not layout(PROPERTIES | dict(pt1_a=1))[0].tables


def test_leaves_go_to_the_first_rows_and_the_others_are_empty():
    lay, names = layout(PROPERTIES)
    (table,) = lay.tables
    # Two walkers: one with 2 leaves, one with none
    fixed = np.array([[1.0], [1.5]])
    leaves = np.array([[1.0, 10.0, 0.5], [2.0, 20.0, 0.7]])
    points = lay.points(
        [fixed, leaves], [np.array([0, 1]), np.array([0, 0])], len(names))
    values = dict(zip(names, points[0]))
    assert [values[f'pt1_a[{i}]'] for i in range(3)] == [0.5, 0.7, 0.0]
    assert values['pt1_r[1]'] == 2.0
    other = dict(zip(names, points[1]))
    assert [other[f'pt1_a[{i}]'] for i in range(3)] == [0.0] * 3
    assert other['pt_a'] == 1.5
