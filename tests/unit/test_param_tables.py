"""
Tests of the tables of parameters (see ParamTable): the blobs of the
mixture traits, as the models that combine traits name their columns.
"""

from gbkfit.model import galaxy_parser
from gbkfit.params import param_tables


def test_mixture_blobs_are_tables_of_their_galaxy():
    # A density and a dispersion mixture: two tables, of the columns of
    # their blobs as the galaxy names them, of the empty blob
    disk = dict(
        type='smdisk', loose=False, tilted=False, rnodes=[0, 5, 10],
        density=dict(ptraits=[
            dict(type='exponential'), dict(type='mixture_gauss', nblobs=3)]),
        kinematics=dict(
            vptraits=dict(type='uniform', direction='tan'),
            dptraits=[
                dict(type='uniform'),
                dict(type='mixture_ggauss', nblobs=2)]),
        emission=dict(type='lines'))
    galaxy = galaxy_parser.load(dict(components=[disk]))
    (blobs, columns), (dblobs, dcolumns) = param_tables(galaxy.pdescs())
    assert columns == {name: f'pt1_{name}' for name in 'rtasqp'}
    assert dcolumns == {name: f'dpt1_{name}' for name in 'rtasbqp'}
    assert blobs is not dblobs
    assert blobs.desc() == 'blob'
    assert blobs.empty() == dict(a=0.0)
    assert galaxy.pdescs()['pt1_r'].size() == 3
